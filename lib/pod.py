import numpy as np

def pod_mode_find(data, config):
    """
    Perform Proper Orthogonal Decomposition (POD) on the input data.
    Args:
        data (np.ndarray): Input data array of shape (n_samples, ...).
    Returns:
        modes (np.ndarray): Modes of the POD decomposition.
        eigVal (np.ndarray): Eigenvalues corresponding to the modes.
        temporal_coeff (np.ndarray): Temporal coefficients of the POD decomposition.
    """
    import time
    if data.shape[0] > 1001:
        flag = 'large'
    orig_shape = data.shape
    config = pod_config(data)
    data = data.reshape(data.shape[0], -1)  # Flatten the spatial dimensions
    
    # if flag == 'large':
    # Low rank svd
    print('Calculating POD modes using low-rank SVD...')
    time_start = time.time()
    modes, eigVal, _ = np.linalg.svd(data.T, full_matrices=False)
    time_end = time.time()
    print(f'POD modes calculated in {time_end - time_start:.2f} seconds.')

    # else:
    #     print('Calculating POD modes using randomized SVD...')
    #     # randomized_svd approach
    #     from sklearn.utils.extmath import randomized_svd
    #     print('Calculating POD modes using randomized SVD...')
    #     print('Calc C matrix', end="")
    #     start = time.time()
    #     C = data.T.dot(data)
    #     C = C / (data.shape[0] - 1)
    #     print(f': {(time.time() - start):.1f}s')
    #     print(f'C shape: {C.shape}')
    #     modes, eigVal, _ = randomized_svd(C, n_components=config['latent_params']['num_modes'], random_state=0)
    

    return modes, eigVal, config


class pod_config:
    def __init__(self, data):
        self.nx = data.shape[1]
        self.ny = data.shape[2]
        self.num_vars = data.shape[3]
        self.num_snaps = data.shape[0]
        self.patch_size = self.nx
        self.nx_t = self.nx
        self.ny_t = self.ny

def pod_decomp(data_path, latent_path, batch_size=1000):
    """
    Perform POD decomposition on the data and save the results.
    Args:
        data_path (str): Path to the input data file.
        latent_path (str): Path to save the latent dofs.
    """
    import h5py
    import time

    total_time = 0.0

    with h5py.File(latent_path, 'r+') as g:
        modes = g['modes'][:]

        with h5py.File(data_path, 'r') as f:
            mean = f['mean'][:]
            num_snaps = f['UV'].shape[0]
            num_batches = num_snaps // batch_size
            if num_snaps % batch_size != 0:
                num_batches += 1
                
            g.create_dataset('dofs', (num_snaps, modes.shape[1]), dtype='float32')
            
            for i in range(num_batches):
                start_time = time.time()
                start = i * batch_size
                end = min((i + 1) * batch_size, num_snaps)
                print(f"Processing batch {i + 1}/{num_batches} ({start}:{end})")
                
                data = f['UV'][start:end, ...] - mean[np.newaxis, ...]
                data = data.reshape(data.shape[0], -1)
                temporal_coeff = np.dot(data, modes)
                end_time = time.time()
                g['dofs'][start:end, :] = temporal_coeff
                
                total_time += end_time - start_time 
            
            print(f"POD decomposition completed in {total_time:.2f} seconds.")
            print(f"Time per snapshot: {total_time / num_snaps:.4f} seconds.")
        
        print(f"POD decomposition completed and saved to {latent_path}")

def pod_recon_long(config, dofs, rec_path, latent_path, batch_size=1000):
    import h5py
    import time
    import sys
    if dofs.dtype == str:
        dof_path = dofs
        with h5py.File(dof_path, 'r') as f:
            dofs = f['dofs'][:]
    
    with h5py.File(latent_path, 'r') as g:
        modes = g['modes'][:,:dofs.shape[-1]]

    num_snaps = dofs.shape[0]
    num_batches = num_snaps // batch_size

    total_time = 0.0

    if num_snaps % batch_size != 0:
        num_batches += 1
    with h5py.File(rec_path, 'w') as rec_file:
        if 'Q_rec' in rec_file.keys():
            del rec_file['Q_rec']
        rec_file.create_dataset('Q_rec', (dofs.shape[0], config.nx_t, config.ny_t, 2), dtype='float32')

        for i in range(num_batches):
            start_time = time.time()
            start = i * batch_size
            end = min((i + 1) * batch_size, num_snaps)
            sys.stdout.write(f"Reconstructing batch {i + 1}/{num_batches} ({start}:{end})")
            sys.stdout.flush()

            time_start = time.time()
            rec_data = np.dot(dofs[start:end, :], modes.T)
            rec_data = rec_data.reshape(rec_data.shape[0], config.nx_t, config.ny_t, 2)
            end_time = time.time()
            total_time += end_time - time_start
            rec_file['Q_rec'][start:end, ...] = rec_data
            time_end = time.time()
            batch_time = time_end - time_start
            sys.stdout.write(f', processed in {batch_time:.2f}s')
            if i+1 != num_batches:
                proj_time = (num_batches - (i + 1)) * batch_time / 60 # in minutes
                # convert to min:sec format
                proj_time_str = f'{int(proj_time)}m {int((proj_time - int(proj_time)) * 60)}s'
                sys.stdout.write(f' -> Proj. time: {proj_time_str}')
            sys.stdout.write('\n')
            sys.stdout.flush()

        print(f'Total reconstruction time (excluding disk I/O): {total_time:.2f}s')
        sys.stdout.write('\n')



def freq_extrapolation(latent_data, dt, new_length):
    import time
    # perform FFT of data along axis 0 to obtain magnitude and phase
    start_time = time.time()
    data_fft = np.fft.fft(latent_data, axis=0)
    end_time = time.time()
    print(f"FFT computed in {end_time - start_time:.2f} seconds.")
    data_fft = data_fft[:latent_data.shape[0]//2 + 1, :]  # keep only positive frequencies

    freq = np.fft.fftfreq(latent_data.shape[0], dt)
    freq = freq[:latent_data.shape[0]//2 + 1] # keep only positive frequencies
    
    num_modes = latent_data.shape[1]
    num_snaps = latent_data.shape[0]

    start_time = time.time()
    # data_extrap_i = \sum_j (mag_i(f_j) * cos(2*pi*f_j*t + phase_i(f_j)) for each i
    data_extrap = np.zeros((new_length, num_modes), dtype=np.complex_)
    t = np.arange(new_length) * dt
    
    for i in range(num_modes):
        mag = np.abs(data_fft[:, i])
        phase = np.unwrap(np.angle(data_fft[:, i]), axis=0)

        for j in range(len(freq)):
            data_extrap[:, i] += 2/(num_snaps) * mag[j] * np.cos(2 * np.pi * freq[j] * t + phase[j])
    
    end_time = time.time()
    print(f"Frequency extrapolation computed in {end_time - start_time:.2f} seconds.")

    return data_extrap.real