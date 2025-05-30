import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['text.usetex'] = True
import pickle
import os
import numpy as np
import h5py
import torch
from scipy.signal import welch, correlate, coherence, correlation_lags, csd
from mpl_toolkits.axes_grid1 import make_axes_locatable

paper_width = 470 # pt
width = paper_width / 72.27 # inches
height = width / 1.618 # inches
# set default font size
plt.rcParams['font.size'] = 10 # Change default font size to 12
plt.rcParams['axes.titlesize'] = 12 # Change axes title font size
plt.rcParams['axes.labelsize'] = 10 # Change axes labels font size
plt.rcParams['xtick.labelsize'] = 10 # Change x-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 10 # Change y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 10 # Change legend font size

def l2_err_norm(true, pred, axis=None):
    """
    Compute the L2 norm between two arrays.
    """
    return np.linalg.norm(true - pred, axis=axis) / np.linalg.norm(true, axis=axis)

def plot_loss(runner):
    with open(runner.paths_bib.model_dir + 'losses.pkl', 'rb') as f:
        results = pickle.load(f)
    size = 0.6
    plt.figure(figsize=(size*width,size*height))
    plt.plot(results['train_losses'], label='Training Loss', color='k', linestyle='-')
    plt.plot(results['test_losses'], label='Test Loss', color='r', linestyle='-.')
    plt.yscale('log')
    plt.title('Losses During Training', pad=16)
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1.025, 0.95),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(runner.paths_bib.fig_dir + 'losses.png', dpi=600)
    plt.close()

def plot_rms(runner, truth, pred):
    """
    Plot the RMS error between the predicted and truth data.
    """
    
    def add_colorbar(ax, im, ticks=None):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, format='%.2f', ticks=ticks)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    time_lag = runner.config['params']['time_lag']
    rms_path = runner.paths_bib.pred_metrics_dir + 'rms.h5'

    if os.path.exists(rms_path):
        with h5py.File(rms_path, 'r') as f:
            rms_true = f['rms_true'][:]
            rms_pred = f['rms_pred'][:]
            rms_error = f['error'][()]
            rms_error_u = f['error_u'][()]
            rms_error_v = f['error_v'][()]
        
    else:
        rms_true = np.sqrt(np.mean(truth[time_lag:]**2, axis=0)).transpose(2,0,1)
        rms_pred = np.sqrt(np.mean(pred[time_lag:]**2, axis=0)).transpose(2,0,1)

        # RMS error 
        rms_error = l2_err_norm(true=rms_true, pred=rms_pred)
        # RMS error on u
        rms_error_u = l2_err_norm(true=rms_true[0], pred=rms_pred[0])
        # RMS error on v
        rms_error_v = l2_err_norm(true=rms_true[1], pred=rms_pred[1])

        # save metrics to paths_bib.pred_metrics_dir
        with h5py.File(rms_path, 'w') as f:
            f.create_dataset('rms_true', data=rms_true)
            f.create_dataset('rms_pred', data=rms_pred)
            f.create_dataset('error', data=rms_error)
            f.create_dataset('error_u', data=rms_error_u)
            f.create_dataset('error_v', data=rms_error_v)

    print(f"RMS error: {100*rms_error:.3f}%")
    print(f"RMS error on u: {100*rms_error_u:.3f}%")
    print(f"RMS error on v: {100*rms_error_v:.3f}%")

    nx = runner.l_config.nx
    ny = runner.l_config.ny
    nx_t = runner.l_config.nx_t
    ny_t = runner.l_config.nx_t

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x = x[:nx_t]
    y = y[:ny_t]
    X, Y = np.meshgrid(x, y)

    RMS_max = np.max(rms_true, axis=(1,2), keepdims=True)
    rms_true_plot = rms_true / RMS_max
    rms_pred_plot = rms_pred / RMS_max
    
    size = 0.75
    
    fig, axs = plt.subplots(1, 2, figsize=(size*width,size*width/2))

    # Example usage with fewer ticks
    ticks = np.linspace(0, 1, 6)

    c1 = axs[0].contourf(X, Y, rms_true_plot[0], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    add_colorbar(axs[0], c1, ticks=ticks)
    axs[0].set_title('True U RMS')
    axs[0].axis('off')
    axs[0].set_aspect('equal')

    c2 = axs[1].contourf(X, Y, rms_pred_plot[0], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    add_colorbar(axs[1], c2, ticks=ticks)
    axs[1].set_title('Predicted U RMS')
    axs[1].axis('off')
    axs[1].set_aspect('equal')

    fig.set_tight_layout(True)
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'rms_u_comparison.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(size*width, size*width/2))
    c1 = axs[0].contourf(X, Y, rms_true_plot[1], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    add_colorbar(axs[0], c1, ticks=ticks)
    axs[0].set_title('True V RMS')
    axs[0].axis('off')
    axs[0].set_aspect('equal')
    c2 = axs[1].contourf(X, Y, rms_pred_plot[1], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    add_colorbar(axs[1], c2, ticks=ticks)
    axs[1].set_title('Predicted V RMS')
    axs[1].axis('off')
    axs[1].set_aspect('equal')

    fig.set_tight_layout(True)
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'rms_v_comparison.png') , dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    
        
    

def plot_tke(runner, truth, pred, idx):
    """
    Plot the TKE of the predicted and truth data over time.
    """
    tke_path = runner.paths_bib.pred_metrics_dir + 'tke.h5'
    
    if os.path.exists(tke_path):
        with h5py.File(tke_path, 'r') as f:
            tke_true = f['tke_true'][:]
            tke_pred = f['tke_pred'][:]
            tke_error = f['error'][()]
    else:
        tke_true = 1/2 * np.sum((truth/512)**2, axis=(1,2,3))
        tke_pred = 1/2 * np.sum((pred/512)**2, axis=(1,2,3))
        # compute L2 error
        tke_error = l2_err_norm(true=tke_true, pred=tke_pred)
        
        # save metrics to paths_bib.pred_metrics_dir
        with h5py.File(runner.paths_bib.pred_metrics_dir + 'tke.h5', 'w') as f:
            f.create_dataset('tke_true', data=tke_true)
            f.create_dataset('tke_pred', data=tke_pred)
            f.create_dataset('error', data=tke_error)

    print(f"TKE error: {100*tke_error:.3f}%")
    t = idx / 100

    size = 0.6
    plt.figure(figsize=(size*width,size*height))
    plt.plot(t, tke_true, label='True TKE', color='k', linestyle='-')
    plt.plot(t, tke_pred, label='Predicted TKE', color='r', linestyle='-.')
    plt.xlabel('Nondimensional time')
    plt.ylabel(r'$\mathrm{TKE} = \frac{1}{2} \sum \mathbf{u}^2$')
    plt.title('Comparison of True and Predicted TKE', pad=16)
    plt.legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.025, 0.95),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'tke_comparison.png'), dpi=600)
    plt.close()

    

    # psd of TKE
    f, Pxx_true = welch(tke_true, fs=100)
    f, Pxx_pred = welch(tke_pred, fs=100)
    plt.figure(figsize=(size*width,size*height))
    plt.loglog(f, Pxx_true, label='True TKE', color='k', linestyle='-')
    plt.loglog(f, Pxx_pred, label='Predicted TKE', color='r', linestyle='-.')
    plt.xlabel('Nondimensional frequency')
    plt.ylabel('PSD(TKE)')
    plt.title('Power Spectral Density of TKE', pad=16)
    plt.legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.025, 0.95),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.tight_layout() #rect=[0, 0, 1, 0.95]
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'tke_psd_comparison.png'), dpi=600)
    plt.close()

    


def plot_PSDs(runner, data_dict):

    psd_results = {}
    # Loop through data types (ground truth and prediction) and points
    for data_type, points in data_dict.items():
        psd_results[data_type] = {}
        for point_name, data in points.items():
            psd_results[data_type][point_name] = {}
            # Compute PSD for the u-component
            f_u, Pxx_u = psd(data[:, 0])
            f_v, Pxx_v = psd(data[:, 1])
            # Combine u and v components into a single array
            Pxx = np.array([Pxx_u, Pxx_v])
            psd_results[data_type][point_name] = Pxx
    psd_results['f'] = f_u
    
    size = 0.75
    fig, axs = plt.subplots(1, 2, figsize=(size*width, size*width/3))
    # Plotting the PSD for U and V component point 1
    axs[0].loglog(psd_results['f'], psd_results['truth']['p1'][0], label='True $u$', color='k', linestyle='-')
    axs[1].loglog(psd_results['f'], psd_results['truth']['p1'][1], label='True $v$', color='k', linestyle='-')
    axs[0].loglog(psd_results['f'], psd_results['pred']['p1'][0], label='Predicted $u$', color='r', linestyle='-.')
    axs[1].loglog(psd_results['f'], psd_results['pred']['p1'][1], label='Predicted $v$', color='r', linestyle='-.')
    axs[0].set_ylabel('PSD($u_{p1}$)')
    axs[1].set_ylabel('PSD($v_{p1}$)')
    axs[0].set_xlabel('Nondimensional frequency')
    axs[1].set_xlabel('Nondimensional frequency')
    # axs[0].legend()
    # axs[1].legend()
    axs[0].grid(visible=True, linestyle='--', linewidth=0.5)
    axs[1].grid(visible=True, linestyle='--', linewidth=0.5)
    axs[1].legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.05, 0.9),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    fig.suptitle('Power Spectral Density of $u$ and $v$ at Point 1')
    fig.tight_layout(rect=[0, 0, 1, 1.15])
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'psd_comparison_p1.png'), dpi=600)
    plt.close()

    # Plotting the PSD for U and V component point 2
    fig, axs = plt.subplots(1, 2, figsize=(size*width, size*width/3))
    axs[0].loglog(psd_results['f'], psd_results['truth']['p2'][0], label='True $u$', color='k', linestyle='-')
    axs[1].loglog(psd_results['f'], psd_results['truth']['p2'][1], label='True $v$', color='k', linestyle='-')
    axs[0].loglog(psd_results['f'], psd_results['pred']['p2'][0], label='Predicted $u$', color='r', linestyle='-.')
    axs[1].loglog(psd_results['f'], psd_results['pred']['p2'][1], label='Predicted $v$', color='r', linestyle='-.')
    axs[0].set_ylabel('PSD($u_{p2}$)')
    axs[1].set_ylabel('PSD($v_{p2}$)')
    axs[0].set_xlabel('Nondimensional frequency')
    axs[1].set_xlabel('Nondimensional frequency')
    axs[1].legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.05, 0.9),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    axs[0].grid(visible=True, linestyle='--', linewidth=0.5)
    axs[1].grid(visible=True, linestyle='--', linewidth=0.5)
    fig.suptitle('Power Spectral Density of $u$ and $v$ at Point 2')
    fig.tight_layout(rect=[0, 0, 1, 1.15])
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'psd_comparison_p2.png'), dpi=600)
    plt.close()

def plot_autocorr(runner, data_dict):
    """
    Plot the autocorrelation of the data.
    """
    size = 1
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    lags, corr_true_u1 = corr(data_dict['truth']['p1'][:,0], data_dict['truth']['p1'][:,0])
    lags, corr_pred_u1 = corr(data_dict['pred']['p1'][:,0], data_dict['pred']['p1'][:,0])

    lags, corr_true_v1 = corr(data_dict['truth']['p1'][:,1], data_dict['truth']['p1'][:,1])
    lags, corr_pred_v1 = corr(data_dict['pred']['p1'][:,1], data_dict['pred']['p1'][:,1])

    lags, corr_true_u2 = corr(data_dict['truth']['p2'][:,0], data_dict['truth']['p2'][:,0])
    lags, corr_pred_u2 = corr(data_dict['pred']['p2'][:,0], data_dict['pred']['p2'][:,0])

    lags, corr_true_v2 = corr(data_dict['truth']['p2'][:,1], data_dict['truth']['p2'][:,1])
    lags, corr_pred_v2 = corr(data_dict['pred']['p2'][:,1], data_dict['pred']['p2'][:,1])

    # Plotting the autocorrelation for U and V component point 1
    lags = lags / 100  # Convert to nondimensional time

    axs[0,0].plot(lags, corr_true_u1, label='True', color='k', linestyle='-')
    axs[0,0].plot(lags, corr_pred_u1, label='Predicted', color='r', linestyle='-.')
    axs[0,1].plot(lags, corr_true_v1, label='True', color='k', linestyle='-')
    axs[0,1].plot(lags, corr_pred_v1, label='Predicted', color='r', linestyle='-.')
    axs[1,0].plot(lags, corr_true_u2, label='True', color='k', linestyle='-')
    axs[1,0].plot(lags, corr_pred_u2, label='Predicted', color='r', linestyle='-.')
    axs[1,1].plot(lags, corr_true_v2, label='True', color='k', linestyle='-')
    axs[1,1].plot(lags, corr_pred_v2, label='Predicted', color='r', linestyle='-.')

    for ax in axs.flat:
        ax.set_xlim(-25, 25)
    axs[0,0].set_title('Point 1 u-component')
    axs[0,1].set_title('Point 1 v-component')
    axs[1,0].set_title('Point 2 u-component')
    axs[1,1].set_title('Point 2 v-component')
    axs[1,0].set_xlabel('Lag')
    axs[1,1].set_xlabel('Lag')
    axs[0,0].set_ylabel('Autocorrelation')
    axs[1,0].set_ylabel('Autocorrelation')
    fig.tight_layout()
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'autocorr_comparison.png'), dpi=600)
    plt.close()



def plot_coherence(runner, data_dict):
    """
    Plot the coherence between the predicted and truth data.
    """
    size = 1
    fig, axs = plt.subplots(1, 2, figsize=(size*width, size*width/3))
    
    # Compute Coherence for U and V component point 1
    f_u, Cxy_u = coher(data_dict['truth']['p1'][:,0], data_dict['pred']['p1'][:,0])
    f_v, Cxy_v = coher(data_dict['truth']['p1'][:,1], data_dict['pred']['p1'][:,1])

    # Plotting the Coherence for U and V component point 1
    axs[0].semilogx(f_u, Cxy_u, label='Coherence', color='k', linestyle='-')
    axs[1].semilogx(f_v, Cxy_v, label='Coherence', color='k', linestyle='-')

    axs[0].set_ylabel('MSC($u_{p1}$)')
    axs[1].set_ylabel('MSC($v_{p1}$)')
    axs[0].set_xlabel('Nondimensional frequency')
    axs[1].set_xlabel('Nondimensional frequency')
    axs[0].set_ylim([0,1])
    axs[1].set_ylim([0,1])
    fig.suptitle('Magnitude Squared Coherence Between Truth and Prediction at Point 1')
    axs[0].grid(visible=True, linestyle='--', linewidth=0.5)
    axs[1].grid(visible=True, linestyle='--', linewidth=0.5)
    
    fig.tight_layout(rect=[0, 0, 1, 1.1])  # Reduce top margin for suptitle
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'coherence_comparison_p1.png'), dpi=600)
    plt.close()

    # Compute Coherence for U and V component point 2
    f_u, Cxy_u = coher(data_dict['truth']['p2'][:,0], data_dict['pred']['p2'][:,0])
    f_v, Cxy_v = coher(data_dict['truth']['p2'][:,1], data_dict['pred']['p2'][:,1])
    # Plotting the Coherence for U and V component point 2
    fig, axs = plt.subplots(1, 2, figsize=(size*width, size*width/3))
    axs[0].semilogx(f_u, Cxy_u, label='Coherence', color='k', linestyle='-')
    axs[1].semilogx(f_v, Cxy_v, label='Coherence', color='k', linestyle='-')
    axs[0].set_ylabel('MSC($u_{p2}$)')
    axs[1].set_ylabel('MSC($v_{p2}$)')
    axs[0].set_xlabel('Nondimensional frequency')
    axs[1].set_xlabel('Nondimensional frequency')
    axs[0].set_ylim([0,1])
    axs[1].set_ylim([0,1])
    fig.suptitle('Magnitude Squared Coherence Between Truth and Prediction at Point 2')
    axs[0].grid(visible=True, linestyle='--', linewidth=0.5)
    axs[1].grid(visible=True, linestyle='--', linewidth=0.5)

    fig.tight_layout(rect=[0, 0, 1, 1.1])  # Reduce top margin for suptitle
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'coherence_comparison_p2.png'), dpi=600)
    plt.close()

def plot_points(runner):
    nx = runner.l_config.nx
    ny = runner.l_config.ny
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    X, Y = np.meshgrid(x, y)

    # find y closest  = 0.112
    y_closest = np.argmin(np.abs(y - 0.112))

    # find x closest to 0.233 and 0.765
    x_closest1 = np.argmin(np.abs(x - 0.233))
    x_closest2 = np.argmin(np.abs(x - 0.765))

    point_1 = (x[x_closest1], y[y_closest])
    point_2 = (x[x_closest2], y[y_closest])

    # Load the mean flow data
    with h5py.File(runner.paths_bib.data_path, 'r') as f:
        mean_flow = f['mean'][:]

    vort_mean = curl(X, Y, mean_flow[..., 0], mean_flow[..., 1])

    vmax = np.max(np.abs(vort_mean))

    # Create a plot that shows the points in the domain
    plt.figure(figsize=(3, 3))
    plt.contourf(X, Y, vort_mean, cmap='seismic', levels=1000, vmin=-vmax, vmax=vmax)
    plt.scatter(point_1[0], point_1[1], color='k', label='Point 1', s=40)
    plt.scatter(point_2[0], point_2[1], color='k', label='Point 2', s=40)
    plt.text(point_1[0], point_1[1]+0.05, f'Point 1', 
            color='k', fontsize=12, va='bottom', ha='center', bbox=dict(facecolor='lightblue', alpha=0.8))
    plt.text(point_2[0], point_2[1]+0.05, f'Point 2', 
            color='k', fontsize=12, va='bottom', ha='center', bbox=dict(facecolor='lightblue', alpha=0.8))
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid()
    plt.axis('equal')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.tight_layout()

    plt.savefig(runner.paths_bib.fig_dir + 'points.png', dpi=600)
    plt.close()

def plot_point_data(runner, data_dict, idx):
    """
    Plot the data at the points of interest.
    """
    size = 0.75
    fig, axs = plt.subplots(2, 2, figsize=(size*width, size*width/2), sharex=True, sharey=True)
    
    t = idx / 100  # Convert to nondimensional time
    # Plotting the data for Point 1
    axs[0,0].plot(t, data_dict['truth']['p1'][:,0], label='True', color='k', linestyle='-')
    axs[0,0].plot(t, data_dict['pred']['p1'][:,0], label='Predicted', color='r', linestyle='-.')
    axs[0,1].plot(t, data_dict['truth']['p1'][:,1], label='True', color='k', linestyle='-')
    axs[0,1].plot(t, data_dict['pred']['p1'][:,1], label='Predicted', color='r', linestyle='-.')
    
    # axs[0,0].set_title('Point 1 $u$-component')
    # axs[0,1].set_title('Point 1 $v$-component')
    
    axs[0,0].set_ylabel('$u_{p1}$')
    axs[0,1].set_ylabel('$v_{p1}$')
    
    # Plotting the data for Point 2
    axs[1,0].plot(t, data_dict['truth']['p2'][:,0], label='True', color='k', linestyle='-')
    axs[1,0].plot(t, data_dict['pred']['p2'][:,0], label='Predicted', color='r', linestyle='-.')
    axs[1,1].plot(t, data_dict['truth']['p2'][:,1], label='True', color='k', linestyle='-')
    axs[1,1].plot(t, data_dict['pred']['p2'][:,1], label='Predicted', color='r', linestyle='-.')
    
    # axs[1,0].set_title('Point 2 $u$-component')
    # axs[1,1].set_title('Point 2 $v$-component')
    
    axs[1,0].set_ylabel('$u_{p2}$')
    axs[1,1].set_ylabel('$v_{p2}$')
    
    axs[1,0].set_xlabel('Nondimensional time')
    axs[1,1].set_xlabel('Nondimensional time')

    for ax in axs.flat:
        # ax.legend(
        #     loc = 'upper center',
        #     # bbox_to_anchor = (0.5, 1),  # center top, above axes
        #     ncol = 2,                      # spread horizontally
        #     frameon = False                # removes legend border
        # ) 
        ax.grid(visible=True, linestyle='--', linewidth=0.5)
        
    axs[0,1].legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.05, 0.9),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    fig.suptitle('Velocity Data at Points of Interest')
    fig.tight_layout(rect=[0, 0, 1, 1.1])  # Adjust layout to make room for suptitle
    # fig.tight_layout()
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'point_data_comparison.png'), dpi=600)
    plt.close()

    # calculate L2 error for each point
    l2_error_p1 = l2_err_norm(true=data_dict['truth']['p1'], pred=data_dict['pred']['p1'])
    l2_error_p2 = l2_err_norm(true=data_dict['truth']['p2'], pred=data_dict['pred']['p2'])
    print(f"L2 error at Point 1: {100*l2_error_p1:.3f}%")
    print(f"L2 error at Point 2: {100*l2_error_p2:.3f}%")




def attention_maps(runner):
    """
    Plot the attention maps for the predicted data.
    """
    time_lag = runner.config['params']['time_lag']
    # get val_indices
    val_indices = runner.val_indices[:time_lag]
    # load dof_u and dof_v
    with h5py.File(runner.paths_bib.latent_path, 'r') as f:
        dof_u = f['dof_u'][val_indices]
        dof_v = f['dof_v'][val_indices]

        initial_input = np.concatenate((dof_u, dof_v), axis=1)
        with open(os.path.join(runner.paths_bib.model_dir, 'dof_scaler.pkl'), 'rb') as f:
            dof_mean, dof_std = pickle.load(f)
        if dof_mean.dtype == torch.float32:
            dof_mean = dof_mean.numpy()
        if dof_std.dtype == torch.float32:
            dof_std = dof_std.numpy()

        initial_input = (initial_input - dof_mean) / dof_std

        # Convert to PyTorch tensor
        initial_input = torch.tensor(initial_input, dtype=torch.float32).to(runner.device)

        _ = runner.model(initial_input[np.newaxis, ...])

        # Retrieve and visualize attention weights
        attn_weights = runner.model.get_attn()
        num_layers = len(attn_weights)
        num_heads = next(iter(attn_weights.values())).shape[1]  # Get the number of heads from the first layer's weights

        fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 2, num_layers * 2), sharex=True, sharey=True)  
        for i, (layer, weights) in enumerate(attn_weights.items()):
            weights = weights.cpu().numpy()
            # weights = weights.squeeze(0)  # Remove the batch dimension
            for j in range(num_heads):
                ax = axes[i, j] if num_layers > 1 else axes[j]
                sns.heatmap(weights[-1,j], cmap="viridis", ax=ax, cbar=False)
                ax.set_title(f"Layer {i+1}, Head {j+1}")
                if i == num_layers - 1:
                    ax.set_xlabel("Key Positions")
                if j == 0:
                    ax.set_ylabel("Query Positions")
                    

        plt.tight_layout()
        plt.savefig(runner.paths_bib.fig_dir + 'attention_weights.png', dpi=600)
        plt.close()


def animate(runner):
    """
    Create an animation of the truth and predicted data.
    """
    import imageio

    time_lag = runner.config['params']['time_lag']
    time_lim = 1000
    nx = runner.l_config.nx
    ny = runner.l_config.ny
    nx_t = runner.l_config.nx_t
    ny_t = runner.l_config.ny_t
    # t = idx / 100  # Convert to nondimensional time

    for pred_file in os.listdir(runner.paths_bib.predictions_dir):
        if pred_file.endswith('.h5') and 'rec' in pred_file:
            print(f"Loading predictions from {pred_file}")
            pred_name = pred_file.replace('rec_', '').replace('_pred.h5', '')
            with h5py.File(os.path.join(runner.paths_bib.predictions_dir, pred_file), 'r') as f:
                num_snaps = f['Q_rec'].shape[0]
                if num_snaps > time_lim:
                    num_snaps = time_lim

                pred = f['Q_rec'][time_lag:num_snaps:2]
                idx = f['idx'][time_lag:num_snaps:2]

            print(f'Loading truth from {runner.paths_bib.data_path}')
            with h5py.File(runner.paths_bib.data_path, 'r') as f:
                mean = f['mean'][:nx_t, :ny_t]
                truth = f['UV'][idx, :nx_t, :ny_t, :] - mean[np.newaxis, ...]
            
            # Root mean squared error normalized by range of truth
            error_plot = 100 * abs(truth - pred) / np.max(np.abs(truth), axis=(0,1,2), keepdims=True)

            Q_plot = truth / np.max(np.abs(truth), axis=(0,1,2), keepdims=True)
            Q_plot_pred = pred / np.max(np.abs(truth), axis=(0,1,2), keepdims=True)

            
            frames = []
            for i, id in enumerate(idx):  # Number of frames
                if i % 50 == 0:
                    print(f'Processing frame {i+1}/{len(idx)}')
                fig, axs = plt.subplots(2,3, figsize=(8,5.5))
                axs[0, 0].imshow(Q_plot[i, :, :, 0], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[0, 0].set_title('True $u$')

                axs[1, 0].imshow(Q_plot[i, :, :, 1], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[1, 0].set_title('True $v$')

                axs[0, 1].imshow(Q_plot_pred[i, :, :, 0], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[0, 1].set_title('Predicted $u$')

                axs[1, 1].imshow(Q_plot_pred[i, :, :, 1], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[1, 1].set_title('Predicted $v$')


                im = axs[0, 2].imshow(error_plot[i, :, :, 0], cmap='RdBu_r', origin='lower', vmin=0, vmax=100)
                
                axs[0, 2].set_title('Error\% $u$')
                axs[0, 2].set_aspect('equal')
                divider = make_axes_locatable(axs[0, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, format='%.0f')
                im.set_clim(vmin=0, vmax=100) # Set color limits for consistency

                cbar.set_label('Error (\%)', rotation=270, labelpad=15)
                im = axs[1, 2].imshow(error_plot[i, :, :, 1], cmap='RdBu_r', origin='lower', vmin=0, vmax=100)
                axs[1, 2].set_title('Error\% $v$')
                axs[1, 2].set_aspect('equal')
                divider = make_axes_locatable(axs[1, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, format='%.0f')
                im.set_clim(vmin=0, vmax=100) # Set color limits for consistency
                cbar.set_label('Error (\%)', rotation=270, labelpad=15)
                
                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)
                plt.tight_layout()
                
                # Save the current frame
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)

            # Save as GIF
            # imageio.mimsave(runner.paths_bib.anim_dir + pred_name + '.gif', frames, fps=30, loop=0)

            # save as mp4
            writer = imageio.get_writer(runner.paths_bib.anim_dir + pred_name + '.mp4', fps=30)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

    




def psd(data, fs = 100):
    """
    Calculate the Power Spectral Density (PSD)
    """
    f, Pxx_u = welch(data, fs)
    return f, Pxx_u
def cpsd(data1, data2, fs = 100):
    """
    Calculate the Cross Power Spectral Density (CPSD)
    """
    f, Cxy = csd(data1, data2, fs)
    return f, Cxy
def corr(data1, data2):
    """
    Calculate the auto or cross-correlation
    """
    corr = correlate(data1, data2, mode='full')
    lags = correlation_lags(len(data1), len(data2), mode='full')
    return lags, corr
def coher(data1, data2, fs = 100):
    """
    Calculate the Coherence
    """
    f, Cxy = coherence(data1, data2, fs)
    return f, Cxy

def curl(x,y,u,v):
    dx = x[0, 1] - x[0, 0]  # Calculate scalar spacing in x-direction
    dy = y[1, 0] - y[0, 0]  # Calculate scalar spacing in y-direction
    # print(u.shape, v.shape)
    dummy, dFx_dy = np.gradient(u, dy, dx, axis=[0,1])  # Note the order of dy and dx
    dFy_dx, dummy = np.gradient(v, dy, dx, axis=[0,1])  # Note the order of dy and dx

    rot = dFy_dx - dFx_dy

    return rot
