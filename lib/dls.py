import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time
import sys
import h5py
from tqdm import tqdm
from scipy.sparse.linalg import factorized
import lib.models as models

def random_patch_sampling(data, patch_size):
    num_patches = 10000
    num_images = 1
    nx = data.shape[0]
    ny = data.shape[1]
    sz = patch_size
    BUFF = 0
    totalsamples = 0
    X = np.zeros((sz ** 2, num_patches))
    
    for i in range(num_images):
        this_image = data

        # Determine how many patches to take
        getsample = num_patches // num_images
        if i == num_images - 1:
            getsample = num_patches - totalsamples

        # Extract patches at random from this image to make data vector X
        for j in range(getsample):
            d1 = BUFF + np.random.randint(0, nx - sz - 2 * BUFF)
            d2 = BUFF + np.random.randint(0, ny - sz - 2 * BUFF)
            
            totalsamples += 1
            temp = this_image[d1:d1 + sz, d2:d2 + sz].reshape(sz ** 2, order='F')
            X[:, totalsamples - 1] = temp - np.mean(temp)

    
    return X

def Modal_decomp_2D(data, patch_size):
    P = random_patch_sampling(data, patch_size)
    local_modes, eigVal, _ = np.linalg.svd(P, full_matrices=False)
    return local_modes, eigVal

def FEM_shape_calculator_2D_ortho_gfemlr(x, y, xpt, ypt):
    sumxpt = np.sum(xpt) / 4
    sumypt = np.sum(ypt) / 4

    dxpt = (-xpt[0] + xpt[1] + xpt[2] - xpt[3]) / 2
    dypt = (ypt[0] + ypt[1] - ypt[2] - ypt[3]) / 2

    zeta_i = [-1, 1, 1, -1]
    eta_i = [1, 1, -1, -1]

    # Inverse transform for parallelogram elements, bilinear shape functions
    zeta = 2 * (x - sumxpt) / dxpt
    eta = 2 * (y - sumypt) / dypt

    N = np.zeros((4,1))
    # shape function values
    for i in range(4):
        N[i] = (1 / 4) * (1 + zeta_i[i] * zeta) * (1 + eta_i[i] * eta)
    return N

def gfem_recon(dof_u, dof_v, config):

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    nskip = config.nskip
    dof_node = config.dof_node # DOFs/node
    dof_elem = config.dof_elem # DOFs/element
    
    # if one dimensional, make 2d
    if len(dof_u.shape) == 1:
        dof_u = dof_u[:, np.newaxis]
        dof_v = dof_v[:, np.newaxis]

    Q_rec_u = np.zeros((config.nx_t, config.ny_t, dof_u.shape[-1]))
    Q_rec_v = np.zeros((config.nx_t, config.ny_t, dof_v.shape[-1]))

    for i in range(config.nx_g-1):
        for j in range(config.ny_g-1):
            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*config.ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)
            # print(lltogl)
            for id in range(dof_u.shape[-1]):
                Q_rec_local_u_vec = config.modemat_local_u @ dof_u[lltogl, id]
                Q_rec_local_v_vec = config.modemat_local_v @ dof_v[lltogl, id]
                
                Q_rec_local_u = Q_rec_local_u_vec.reshape((nskip+1, nskip+1), order='F')
                Q_rec_local_v = Q_rec_local_v_vec.reshape((nskip+1, nskip+1), order='F')

                Q_rec_u[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_u
                Q_rec_v[config.sample_x[i]:config.sample_x[i+1]+1, config.sample_y[j]:config.sample_y[j+1]+1, id] = Q_rec_local_v
    Q_rec = np.zeros((dof_u.shape[-1], config.nx_t, config.ny_t, 2))
    Q_rec[:, :, :, 0] = Q_rec_u.transpose(2,0,1)
    Q_rec[:, :, :, 1] = Q_rec_v.transpose(2,0,1)
    return Q_rec

class dls_Config:
    def __init__(self, data, patch_size, num_modes, modemat_local_u, modemat_local_v):
        self.nx = data.shape[1]
        self.ny = data.shape[2]
        self.num_snaps = data.shape[3]
        self.patch_size = patch_size
        self.num_modes = num_modes
        self.nskip = (patch_size - 1) // 2
        self.nskip_sample = patch_size - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.sample_x = range(0, self.nx, self.nskip)
        self.sample_y = range(0, self.ny, self.nskip)
        self.nx_t = max(self.sample_x) + 1
        self.ny_t = max(self.sample_y) + 1
        self.nx_g = len(self.sample_x)
        self.ny_g = len(self.sample_y)
        self.num_gfem_nodes = self.nx_g * self.ny_g
        self.dof_node = num_modes + 1
        self.dof_elem = 4 * self.dof_node
        self.modemat_local_u = modemat_local_u
        self.modemat_local_v = modemat_local_v
        self.compression_ratio = data.shape[0]*self.num_snaps*self.nx*self.ny / (data.shape[0]*self.num_snaps*self.dof_node + data.shape[0] * num_modes * self.patch_size**2 )


def gfem_2d_long(data_path: str, field_name: str, latent_file: str, patch_size: int, num_modes: int, batch_size: int = 2500):
    with h5py.File(data_path, 'r') as f:
        num_snaps = f[field_name].shape[0]
        nx = f[field_name].shape[1]
        ny = f[field_name].shape[2]
        num_vars = f[field_name].shape[3]
        mode_data = f[field_name][0,:,:,:].transpose(2,0,1) - f['mean'][:].transpose(2,0,1)
        print('shape of mode data: ', mode_data.shape)
        print('number of snapshots: ', num_snaps)
        print('number of batches: ', num_snaps // batch_size)
        print('nx: ', nx)
        print('ny: ', ny)
        print('num_vars: ', num_vars)


    grid_x = np.linspace(1, nx, nx)
    grid_y = np.linspace(1, ny, ny)
    [grid_x, grid_y] = np.meshgrid(grid_x, grid_y)
    nskip = (patch_size - 1) // 2
    nskip_sample = patch_size - 1
    mid_pt = 1 + nskip_sample // 2

    # GFEM grid points
    sample_x = range(0, nx, nskip)
    sample_y = range(0, ny, nskip)

    # Truncated grid size
    nx_t = max(sample_x)+1
    ny_t = max(sample_y)+1

    # GFEM grid size
    nx_g = len(sample_x)
    ny_g = len(sample_y)

    num_gfem_nodes = nx_g * ny_g # total number of nodes in the GFEM grid
    dof_node = num_modes+1 # DOFs/node
    dof_elem = 4 * dof_node # DOFs/element


    # Compute local modes
    local_modes_u, eigVal = Modal_decomp_2D(mode_data[0], patch_size)
    local_modes_v, eigVal = Modal_decomp_2D(mode_data[1], patch_size)
    # print(local_modes.shape)
    mode_grid_u = local_modes_u[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')
    mode_grid_v = local_modes_v[:, :num_modes].reshape(patch_size, patch_size, num_modes, order='F')

    # Mode grid components for the four quadrants
    F1 = list(range(0, mid_pt))
    F2 = list(range(mid_pt-1, nskip_sample + 1))
    F3 = list(range(0, mid_pt))
    F4 = list(range(mid_pt-1, nskip_sample + 1))

    modes_grid_1_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_u = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_u = np.zeros((mid_pt, mid_pt, num_modes))

    modes_grid_1_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_2_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_3_comp_v = np.zeros((mid_pt, mid_pt, num_modes))
    modes_grid_4_comp_v = np.zeros((mid_pt, mid_pt, num_modes))

    for i in range(num_modes):
        modes_grid_1_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_u[:, :, i] = mode_grid_u[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_u[:, :, i] = mode_grid_u[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

        modes_grid_1_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_2_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F4[0]:F4[-1]+1, i]
        modes_grid_3_comp_v[:, :, i] = mode_grid_v[F2[0]:F2[-1]+1, F3[0]:F3[-1]+1, i]
        modes_grid_4_comp_v[:, :, i] = mode_grid_v[F1[0]:F1[-1]+1, F3[0]:F3[-1]+1, i]

    modes_vec_1_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_u = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_u = np.zeros((mid_pt ** 2, num_modes))

    modes_vec_1_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_2_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_3_comp_v = np.zeros((mid_pt ** 2, num_modes))
    modes_vec_4_comp_v = np.zeros((mid_pt ** 2, num_modes))

    for i in range(num_modes):
        modes_vec_1_comp_u[:, i] = modes_grid_1_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_u[:, i] = modes_grid_2_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_u[:, i] = modes_grid_3_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_u[:, i] = modes_grid_4_comp_u[:, :, i].reshape((mid_pt) ** 2, order='F')

        modes_vec_1_comp_v[:, i] = modes_grid_1_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_2_comp_v[:, i] = modes_grid_2_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_3_comp_v[:, i] = modes_grid_3_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
        modes_vec_4_comp_v[:, i] = modes_grid_4_comp_v[:, :, i].reshape((mid_pt) ** 2, order='F')
    
    i = 1
    j = 1

    M_local_u = np.zeros((dof_elem, dof_elem))
    M_local_v = np.zeros((dof_elem, dof_elem))
    
    # x, y locations of the GFEM element nodes
    x1 = grid_x[i*nskip,       (j-1)*nskip]
    x2 = grid_x[i*nskip,       j*nskip]
    x3 = grid_x[(i-1)*nskip,   j*nskip]
    x4 = grid_x[(i-1)*nskip,   (j-1)*nskip]

    y1 = grid_y[i*nskip,       (j-1)*nskip]
    y2 = grid_y[i*nskip,       j*nskip]
    y3 = grid_y[(i-1)*nskip,   j*nskip]
    y4 = grid_y[(i-1)*nskip,   (j-1)*nskip]


    # Combining x, y nodal coordinates into vector form
    xpt = [x1, x2, x3, x4]
    ypt = [y1, y2, y3, y4]

    N1 = np.zeros((nskip+1)**2)
    N2 = np.zeros((nskip+1)**2)
    N3 = np.zeros((nskip+1)**2)
    N4 = np.zeros((nskip+1)**2)

    for kx in range(nskip+1):
        indx = (i-1)*nskip + kx
        for ky in range(nskip+1):
            indy = (j-1)*nskip + ky
            x_val = grid_x[indy,indx]
            y_val = grid_y[indy,indx]

            # shape functions over the grid points

            iind = ky*(nskip+1) + kx

            N = FEM_shape_calculator_2D_ortho_gfemlr(x_val, y_val, xpt, ypt)

            N1[iind] = N[0][0]
            N2[iind] = N[1][0]
            N3[iind] = N[2][0]
            N4[iind] = N[3][0]

    Wt = np.ones((nskip+1, nskip+1))
    Wt[1:-1,0] = 1/2
    Wt[1:-1,-1] = 1/2
    Wt[0,1:-1] = 1/2
    Wt[-1,1:-1] = 1/2
    Wt[0,0] = 1/4
    Wt[0,-1] = 1/4
    Wt[-1,0] = 1/4
    Wt[-1,-1] = 1/4

    Wt_vec = Wt.reshape((nskip+1)**2, order='F')

    modemat_local_u = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_u,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_u,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_u,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_u
    ])
    
    modemat_local_v = np.hstack([
        N1[:, np.newaxis],
        N1[:, np.newaxis] * modes_vec_3_comp_v,
        N2[:, np.newaxis],
        N2[:, np.newaxis] * modes_vec_4_comp_v,
        N3[:, np.newaxis],
        N3[:, np.newaxis] * modes_vec_1_comp_v,
        N4[:, np.newaxis],
        N4[:, np.newaxis] * modes_vec_2_comp_v
    ])

    modemat_local_u_wt = np.zeros_like(modemat_local_u)
    modemat_local_v_wt = np.zeros_like(modemat_local_v)

    for kk in range(modemat_local_u.shape[1]):
        modemat_local_u_wt[:, kk] = modemat_local_u[:, kk] * Wt_vec
        modemat_local_v_wt[:, kk] = modemat_local_v[:, kk] * Wt_vec

    # local mass matrix
    M_local_u = modemat_local_u_wt.T @ modemat_local_u
    M_local_v = modemat_local_v_wt.T @ modemat_local_v

    M_u = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))
    M_v = lil_matrix((num_gfem_nodes * dof_node, num_gfem_nodes * dof_node))

    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    print('Constructing global M GFEM matrix')

    for i in range(nx_g-1):
        for j in range(ny_g-1):
            lltogl = np.zeros(dof_elem, dtype=int)
            for lindx in range(4):
                indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                indx_dof_end = indx_dof_start + dof_node

                lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

            M_u[np.ix_(lltogl, lltogl)] = M_u[np.ix_(lltogl, lltogl)] + M_local_u
            M_v[np.ix_(lltogl, lltogl)] = M_v[np.ix_(lltogl, lltogl)] + M_local_v



    print('M constructed')

    print('Prefactorizing M')
    # Convert lilmatrix to csr matrix
    M_u = M_u.tocsc()
    M_v = M_v.tocsc()
    # Pre-factorize the matrices for efficiency
    solve_M_u = factorized(M_u)
    solve_M_v = factorized(M_v)
    print('Done prefactorizing M')

    # create h5 dataset for dof of shape (num_gfem_nodes*dof_node, num_snaps)
    dof_file = h5py.File(latent_file, 'w')
    dof_file.create_dataset('dof_u', (num_snaps, num_gfem_nodes * dof_node), dtype='float32')
    dof_file.create_dataset('dof_v', (num_snaps, num_gfem_nodes * dof_node), dtype='float32')

    dof_u = np.zeros((num_gfem_nodes * dof_node, batch_size))
    dof_v = np.zeros((num_gfem_nodes * dof_node, batch_size))


    print('Looping through snapshots, solving for dofs')
    with h5py.File(data_path, 'r') as f:
        loops = num_snaps // batch_size
        if num_snaps % batch_size != 0:
            loops += 1
        for i in tqdm(range(loops)):
            snap_start = i * batch_size
            snap_end = (i + 1) * batch_size
            if snap_end > num_snaps:
                snap_end = num_snaps
            batch_size = snap_end - snap_start
            u_mean = f['mean'][:, :, 0]
            v_mean = f['mean'][:, :, 1]
            Q_grid_u = f[field_name][snap_start:snap_end,:,:,0]
            Q_grid_v = f[field_name][snap_start:snap_end,:,:,1]
            Q_grid_u = Q_grid_u.transpose(1,2,0) - u_mean[:,:, np.newaxis]
            Q_grid_v = Q_grid_v.transpose(1,2,0) - v_mean[:,:, np.newaxis]
    
            L_u = np.zeros((num_gfem_nodes * dof_node, batch_size))
            L_v = np.zeros((num_gfem_nodes * dof_node, batch_size))

            for i in range(nx_g-1):
                for j in range(ny_g-1):
                    L_local_u = np.zeros((dof_elem, batch_size))
                    L_local_v = np.zeros((dof_elem, batch_size))

                    lltogl = np.zeros(dof_elem, dtype=int)
                    for lindx in range(4):
                        indx_dof_start = ((i+IJK[lindx, 0])*ny_g + (j+IJK[lindx, 1]))*dof_node
                        indx_dof_end = indx_dof_start + dof_node

                        lltogl[lindx*dof_node: (lindx+1)*dof_node] = np.arange(indx_dof_start, indx_dof_end)

                    indx_cell = i * nskip
                    indy_cell = j * nskip

                    Q_local_u = Q_grid_u[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, :]
                    Q_local_v = Q_grid_v[indx_cell:indx_cell+nskip+1, indy_cell:indy_cell+nskip+1, :]

                    Q_local_u_vec = np.zeros(((nskip+1)**2, batch_size))
                    Q_local_v_vec = np.zeros(((nskip+1)**2, batch_size))

                    for kx in range(nskip+1):
                        for ky in range(nskip+1):
                            iind = ky*(nskip+1) + kx
                            
                            Q_local_u_vec[iind, :] = Q_local_u[kx, ky, :]
                            Q_local_v_vec[iind, :] = Q_local_v[kx, ky, :]

                    L_local_u = modemat_local_u_wt.T @ Q_local_u_vec
                    L_local_v = modemat_local_v_wt.T @ Q_local_v_vec

                    L_u[lltogl,:] = L_u[lltogl, :] + L_local_u
                    L_v[lltogl,:] = L_v[lltogl, :] + L_local_v
            
            dof_u = solve_M_u(L_u)
            dof_v = solve_M_v(L_v)

            dof_file['dof_u'][snap_start:snap_end] = dof_u.T
            dof_file['dof_v'][snap_start:snap_end] = dof_v.T
            

    print('Done solving for dof')


    config = dls_long_Config(data_path, field_name, patch_size, num_modes, modemat_local_u, modemat_local_v)

    return config


class dls_long_Config:
    def __init__(self, data_path, field_name, patch_size, num_modes, modemat_local_u, modemat_local_v):
        with h5py.File(data_path, 'r') as f:
            self.num_snaps = f[field_name].shape[0]
            self.nx = f[field_name].shape[1]
            self.ny = f[field_name].shape[2]
            self.num_vars = f[field_name].shape[3]
        self.patch_size = patch_size
        self.num_modes = num_modes
        self.nskip = (patch_size - 1) // 2
        self.nskip_sample = patch_size - 1
        self.mid_pt = 1 + self.nskip_sample // 2
        self.sample_x = range(0, self.nx, self.nskip)
        self.sample_y = range(0, self.ny, self.nskip)
        self.nx_t = max(self.sample_x) + 1
        self.ny_t = max(self.sample_y) + 1
        self.nx_g = len(self.sample_x)
        self.ny_g = len(self.sample_y)
        self.num_gfem_nodes = self.nx_g * self.ny_g
        self.dof_node = num_modes + 1
        self.dof_elem = 4 * self.dof_node
        self.modemat_local_u = modemat_local_u
        self.modemat_local_v = modemat_local_v
        self.compression_ratio = self.num_vars*self.num_snaps*self.nx*self.ny / (self.num_vars*self.num_snaps*self.dof_node + self.num_vars * num_modes * self.patch_size**2 )
        


def gfem_recon_long(rec_path, config, dof_u=None, dof_v=None, batch_size=100):
    if dof_u.dtype == str:
        dof_path = dof_u
        with h5py.File(dof_path, 'r') as f:
            dof_u = f['dof_u'][:].T
            dof_v = f['dof_v'][:].T
            
    num_snaps = dof_u.shape[1]
    num_batches = num_snaps // batch_size
    if num_snaps % batch_size != 0:
        num_batches += 1


    IJK = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    nskip = config.nskip
    dof_node = config.dof_node  # DOFs/node
    dof_elem = config.dof_elem  # DOFs/element

    with h5py.File(rec_path, 'w') as rec_file:
        if 'Q_rec' in rec_file.keys():
            del rec_file['Q_rec']
        rec_file.create_dataset('Q_rec', (dof_u.shape[-1], config.nx_t, config.ny_t, 2), dtype='float32')

        for id in range(num_batches):
            

            snap_start = id * batch_size
            snap_end = (id + 1) * batch_size
            if snap_end >= num_snaps:
                snap_end = num_snaps
            batch_size = snap_end - snap_start
            time_start = time.time()
            sys.stdout.write(f'Processing batch {id+1}/{num_batches}, batch size: {batch_size}')
            sys.stdout.flush()

            Q_rec_u = np.zeros((config.nx_t, config.ny_t, batch_size))
            Q_rec_v = np.zeros((config.nx_t, config.ny_t, batch_size))

            for i in range(config.nx_g - 1):
                for j in range(config.ny_g - 1):
                    lltogl = np.zeros(dof_elem, dtype=int)
                    for lindx in range(4):
                        indx_dof_start = ((i + IJK[lindx, 0]) * config.ny_g + (j + IJK[lindx, 1])) * dof_node
                        indx_dof_end = indx_dof_start + dof_node

                        lltogl[lindx * dof_node: (lindx + 1) * dof_node] = np.arange(indx_dof_start, indx_dof_end)

                    Q_rec_local_u_vec = config.modemat_local_u @ dof_u[lltogl, snap_start:snap_end]
                    Q_rec_local_v_vec = config.modemat_local_v @ dof_v[lltogl, snap_start:snap_end]

                    Q_rec_local_u = Q_rec_local_u_vec.reshape((nskip + 1, nskip + 1, batch_size), order='F')
                    Q_rec_local_v = Q_rec_local_v_vec.reshape((nskip + 1, nskip + 1, batch_size), order='F')

                    Q_rec_u[config.sample_x[i]:config.sample_x[i + 1] + 1, config.sample_y[j]:config.sample_y[j + 1] + 1] = Q_rec_local_u
                    Q_rec_v[config.sample_x[i]:config.sample_x[i + 1] + 1, config.sample_y[j]:config.sample_y[j + 1] + 1] = Q_rec_local_v

            rec_file['Q_rec'][snap_start:snap_end, :, :, 0] = Q_rec_u.transpose(2,0,1)
            rec_file['Q_rec'][snap_start:snap_end, :, :, 1] = Q_rec_v.transpose(2,0,1)

            time_end = time.time()
            batch_time = time_end - time_start
            sys.stdout.write(f', processed in {batch_time:.2f}s')
            if id+1 != num_batches:
                proj_time = (num_batches - (id + 1)) * batch_time / 60 # in minutes
                # convert to min:sec format
                proj_time_str = f'{int(proj_time)}m {int((proj_time - int(proj_time)) * 60)}s'
                sys.stdout.write(f' -> Proj. time: {proj_time_str}')
            sys.stdout.write('\n')
            sys.stdout.flush()
        sys.stdout.write('\n')

def latent_eval(runner):
    print(f"{'#'*20}\t{'Evaluating latent...':<20}\t{'#'*20}")
    import math
    from lib.plotting import l2_err_norm, curl
    import torch 
    import pickle

    # pull the latent space from h5 file
    eval_length_max = 2500
    with h5py.File(runner.paths_bib.latent_path, 'r') as f:
        if runner.config['latent_type'] == 'dls':
            latent_length = f['dof_u'].shape[0]
            print('latent length:', latent_length)
            if latent_length > eval_length_max:
                latent_length = eval_length_max
            dof_u = f['dof_u'][:latent_length]
            dof_v = f['dof_v'][:latent_length]
            print('dof_u shape:', dof_u.shape)
            print('dof_v shape:', dof_v.shape)
            # reconstruct the data using the gfem_recon function
            print('Reconstructing data...')
            Q_rec = gfem_recon(dof_u=dof_u.T, dof_v=dof_v.T, config=runner.l_config)
            print('Data shape:', Q_rec.shape)
            print('Data reconstructed.')
        elif runner.config['latent_type'] == 'pod':
            latent_length = f['dofs'].shape[0]
            if latent_length > eval_length_max:
                latent_length = eval_length_max
            dofs = f['dofs'][:latent_length, :runner.config['latent_params']['num_modes']]
            modes = f['modes'][:, :runner.config['latent_params']['num_modes']]
            runner.l_config.num_modes = runner.config['latent_params']['num_modes']
            # reconstruct the data using modes dot dofs
            print('Reconstructing data...')
            Q_rec = np.dot(dofs, modes.T).reshape((latent_length, runner.l_config.nx_t, runner.l_config.ny_t, 2))
            print('Data shape:', Q_rec.shape)
            print('Data reconstructed.')

            # save images of the modes to figs/latent_modes/
            from matplotlib import pyplot as plt
            import os
            os.makedirs(runner.paths_bib.fig_dir + 'latent_modes/', exist_ok=True)
            print('Saving POD modes...')
            for i in range(0, runner.config['latent_params']['num_modes']):
                
                pod_mode = modes[:, i].reshape((runner.l_config.nx_t, runner.l_config.ny_t, 2))
                umax = np.max(np.abs(pod_mode[:,:,0]))
                vmax = np.max(np.abs(pod_mode[:,:,1]))
                fig, ax = plt.subplots(1,2, figsize=(8,4))
                im0 = ax[0].imshow(pod_mode[:,:,0], cmap='seismic', vmin=-umax, vmax=umax, origin='lower', interpolation='bessel')
                ax[0].set_title(f'POD Mode {i+1} - u')
                im1 = ax[1].imshow(pod_mode[:,:,1], cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', interpolation='bessel')
                ax[1].set_title(f'POD Mode {i+1} - v')
                
                plt.savefig(runner.paths_bib.fig_dir + f'latent_modes/pod_mode_{i+1}.png', dpi=600)
                plt.close()

        elif runner.config['latent_type'] == 'bvae':
            from matplotlib import pyplot as plt
            import os
            
            latent_dim = runner.config['latent_params']['latent_dim']
            print('Getting latent data')
            latent_length = f['dofs'].shape[0]
            if latent_length > eval_length_max:
                latent_length = eval_length_max
            dofs = f['dofs'][:latent_length, :latent_dim]
            print('dofs shape:', dofs.shape)

            print('Loading BVAE model')
            data_shape = [runner.l_config.num_vars, runner.l_config.nx_t, runner.l_config.ny_t]
            bvae = models.bvae_model(data_shape, runner.config)
            bvae.load_state_dict(torch.load(runner.paths_bib.latent_model_path, weights_only=True))
            bvae.to(runner.device)

            bvae.eval()

            print('Calculating BVAE mode order and cumulative energy...')

            if latent_dim <= 10:
                order, Ecum = models.bvae_mode_order(bvae, runner.paths_bib.data_path, runner.paths_bib.latent_path, runner.config, runner.device)

                for i in range(len(Ecum)):
                    print(f'BVAE mode {i+1}, Cumulative Energy: {Ecum[i]:.4f}')
            else:
                order = list(range(latent_dim))
                print('Latent dimension > 10, skipping mode order calculation.')

            print('Reconstructing data...')

            dofs = torch.tensor(dofs, dtype=torch.float32).to(runner.device)
            Q_rec = models.bvae_decode(bvae, dofs, runner.device)
            Q_rec = Q_rec.cpu().detach().numpy().transpose(0,2,3,1)
            # rescale the data to the original mean and std
            scaler_path = runner.paths_bib.latent_dir + 'latent_scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    mean, std = pickle.load(f)
                Q_rec = models.denormalize_data(Q_rec, mean, std)
                
                
            print('Data shape:', Q_rec.shape)
            print('Data reconstructed.')

            bvae_modes = np.zeros((latent_dim, Q_rec.shape[1], Q_rec.shape[2], Q_rec.shape[3]))
            os.makedirs(runner.paths_bib.fig_dir + 'latent_modes/', exist_ok=True)

            print('Saving BVAE modes...')
            for i, ind in enumerate(order):
                z = torch.zeros(latent_dim)
                z[ind] = 1
                z = z.to(runner.device)
                bvae_mode = models.bvae_decode(bvae, z.unsqueeze(0), runner.device)
                bvae_mode = bvae_mode.cpu().detach().numpy().squeeze().transpose(1,2,0)
                bvae_modes[i] = bvae_mode
                # Make a figure of the mode and save to figs/latent_modes/
                umax = np.max(np.abs(bvae_mode[:,:,0]))
                vmax = np.max(np.abs(bvae_mode[:,:,1]))
                fig, ax = plt.subplots(1,2, figsize=(8,4))
                im0 = ax[0].imshow(bvae_mode[:,:,0], cmap='seismic', vmin=-umax, vmax=umax, origin='lower', interpolation='bessel')
                ax[0].set_title(f'BVAE Mode {i+1} - u')
                im1 = ax[1].imshow(bvae_mode[:,:,1], cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', interpolation='bessel')
                ax[1].set_title(f'BVAE Mode {i+1} - v')
                
                plt.savefig(runner.paths_bib.fig_dir + f'latent_modes/bvae_mode_{i+1}.png', dpi=600)
                plt.close()

            # # Plot difference between BVAE mode and POD mode if POD modes are available
            # pod_path = 'results/' + runner.config['data_name'] + '/pod/'
            # if os.path.exists(pod_path + 'latent_coeff.h5'):
            #     with h5py.File(pod_path + 'latent_coeff.h5', 'r') as f:
            #         if 'modes' in f.keys():
            #             pod_modes = f['modes'][:, :latent_dim]
            #             for i in range(5):
            #                 pod_mode = pod_modes[:, i+1].reshape((runner.l_config.nx_t, runner.l_config.ny_t, 2))
            #                 print(f'POD mode {i+1} max:', np.max(np.abs(pod_mode)))
            #                 pod_mode = pod_mode / np.max(np.abs(pod_mode.flatten()))
            #                 bvae_mode = bvae_modes[i] / np.max(np.abs(bvae_modes[i].flatten()))
            #                 diff_mode = bvae_mode - pod_mode
            #                 umax = np.max(np.abs(diff_mode[:,:,0]))
            #                 vmax = np.max(np.abs(diff_mode[:,:,1]))
            #                 fig, ax = plt.subplots(1,2, figsize=(8,4))
            #                 im0 = ax[0].imshow(diff_mode[:,:,0], cmap='Greys', vmin=-umax, vmax=umax, origin='lower', interpolation='bessel')
            #                 ax[0].set_title(f'BVAE - POD Mode {i+1} - u')
            #                 im1 = ax[1].imshow(diff_mode[:,:,1], cmap='Greys', vmin=-vmax, vmax=vmax, origin='lower', interpolation='bessel')
            #                 ax[1].set_title(f'BVAE - POD Mode {i+1} - v')
                            
            #                 plt.savefig(runner.paths_bib.fig_dir + f'diff_latent_modes/bvae_pod_diff_mode_{i+1}.png', dpi=600)
            #                 plt.close()

    if runner.config['latent_type'] == 'bvae':
        with h5py.File(runner.paths_bib.latent_path, 'a') as f:
            if 'modes' in f.keys():
                del f['modes']
            f.create_dataset('modes', data=bvae_modes)

        with h5py.File(runner.paths_bib.data_path, 'r') as f:
            mean = f['mean'][:]
            Q = f['UV'][:latent_length] - mean[np.newaxis, ...]

            nx_t = Q.shape[1]
            ny_t = Q.shape[2]
            x = np.linspace(0, 1, nx_t)
            y = np.linspace(0, 1, ny_t)

            X, Y = np.meshgrid(x, y)

        # save reconstruction comparison to ground truth for first snapshot
        fig, ax = plt.subplots(2, 2, figsize=(8,8))
        vmax = np.max(np.abs(Q[0,:,:,:]))
        im0 = ax[0,0].imshow(Q[0,:,:,0], cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', interpolation='bessel')
        ax[0,0].set_title('Ground Truth - u')
        im1 = ax[0,1].imshow(Q[0,:,:,1], cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', interpolation='bessel')
        ax[0,1].set_title('Ground Truth - v')
        im2 = ax[1,0].imshow(Q_rec[0,:,:,0], cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', interpolation='bessel')
        ax[1,0].set_title('Reconstruction - u')
        im3 = ax[1,1].imshow(Q_rec[0,:,:,1], cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower', interpolation='bessel')
        ax[1,1].set_title('Reconstruction - v')
        plt.colorbar(im3, ax=ax, orientation='vertical', fraction=.1, shrink=0.8)
        plt.suptitle('BVAE Reconstruction vs Ground Truth (Snapshot 1)')
        plt.savefig(runner.paths_bib.fig_dir + 'bvae_reconstruction_comparison.png', dpi=600)
        plt.close()



    
    if runner.config['latent_type'] in ['dls', 'pod']:
        nx = runner.l_config.nx
        ny = runner.l_config.ny
        nx_t = runner.l_config.nx_t
        ny_t = runner.l_config.ny_t

        # point probe info
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        x = x[:nx_t]
        y = y[:ny_t]

        X, Y = np.meshgrid(x, y)

        # load the ground truth data
        print('Loading ground truth data...')
        with h5py.File(runner.paths_bib.data_path, 'r') as f:
            mean = f['mean'][:nx_t, :ny_t, :]
            Q = f['UV'][:latent_length, :nx_t, :ny_t, :] - mean[np.newaxis, ...]
        print('Q shape:', Q.shape)
        print('Data loaded.')

        # calculate the compression ratio
        M = math.prod(Q.shape) 
        T = latent_length
        d = 2
        p = runner.l_config.patch_size
        m = runner.l_config.num_modes
        if runner.config['latent_type'] == 'dls':
            n = runner.l_config.num_gfem_nodes
            CR = M / ( d * T * n * (m+1) + d * m * p**2)
        else: 
            CR = M / (T * m + m * p**2)
        

        print(f'Compression Ratio: {CR}')

    L2_error = np.sqrt(np.sum((Q-Q_rec)**2, axis=(1,2,3))) / np.sqrt(np.sum(Q**2, axis=(1,2,3)))
    L2_error_mean = L2_error.mean()
    print(f'L2 Error Mean: {100*L2_error_mean:.4f}%')


    # vorticity reconstruction error
    vort = curl(X, Y , Q[..., 0].transpose(1,2,0), Q[..., 1].transpose(1,2,0))
    vort = vort.transpose(2,0,1)
    vort_rec = curl(X, Y , Q_rec[..., 0].transpose(1,2,0), Q_rec[..., 1].transpose(1,2,0))
    vort_rec = vort_rec.transpose(2,0,1)

    vort_error = l2_err_norm(true=vort, pred=vort_rec, axis=(1, 2))
    vort_error_mean = vort_error.mean()

    print(f'Vorticity Error Mean: {100*vort_error_mean:.4f}%')

    # tke reconstruction error
    tke = 0.5 * np.sum(Q**2, axis=(1,2,3))
    tke_rec = 0.5 * np.sum(Q_rec**2, axis=(1,2,3))

    tke_error = l2_err_norm(true=tke, pred=tke_rec)

    print(f'TKE Error: {100*tke_error:.4f}%')

    RMS = np.sqrt(np.mean(Q**2, axis=0))
    RMS_rec = np.sqrt(np.mean(Q_rec**2, axis=0))
    RMS_error = np.sqrt(np.sum((RMS - RMS_rec)**2)) / np.sqrt(np.sum(RMS**2))

    print(f'RMS Error: {100*RMS_error:.4f}%')

    print('Saving error metrics to latent_path...')

    # save error metrics to latent_path h5
    with h5py.File(runner.paths_bib.latent_path, 'a') as f:
        if 'CR' not in f.keys() and runner.config['latent_type'] in ['dls', 'pod']:
            f.create_dataset('CR', data=CR)
        if 'L2_error' not in f.keys():
            f.create_dataset('L2_error', data=L2_error)
        if 'L2_error_mean' not in f.keys():
            f.create_dataset('L2_error_mean', data=L2_error_mean)
        if 'vort_error' not in f.keys():
            f.create_dataset('vort_error', data=vort_error)
        if 'vort_error_mean' not in f.keys():
            f.create_dataset('vort_error_mean', data=vort_error_mean)
        if 'tke_error' not in f.keys():
            f.create_dataset('tke_error', data=tke_error)
        if 'RMS_error' not in f.keys():
            f.create_dataset('RMS_error', data=RMS_error)

    
    
    
    

    