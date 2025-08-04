import matplotlib
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
from matplotlib.ticker import FuncFormatter

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
plt.rcParams['figure.constrained_layout.use'] = True

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
    # plt.tight_layout()
    plt.savefig(runner.paths_bib.fig_dir + 'losses.png', dpi=600)
    plt.close()

def plot_rms(runner, pred_path, eval_idx, true_idx):
    """
    Plot the RMS error between the predicted and truth data.
    """

    def add_colorbar(ax, im, ticks=None):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, format='%.2f', ticks=ticks, shrink=0.1)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    with h5py.File(pred_path, 'r') as f:
        rms_true = f['rms_true'][:]
        rms_pred = f['rms_pred'][:]
    # RMS error 
    rms_error = l2_err_norm(true=rms_true, pred=rms_pred)
    # RMS error on u
    rms_error_u = l2_err_norm(true=rms_true[0], pred=rms_pred[0])
    # RMS error on v
    rms_error_v = l2_err_norm(true=rms_true[1], pred=rms_pred[1])

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

    ticks = np.linspace(0, 1, 6)
    
    fig, axs = plt.subplots(1, 2, figsize=(size*width,size*width/2))
    c1 = axs[0].contourf(X, Y, rms_true_plot[0], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[0].set_title('True U RMS')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlim(0, 1.0)
    axs[0].set_ylim(0, 1.0)
    axs[0].set_aspect('equal')

    c2 = axs[1].contourf(X, Y, rms_pred_plot[0], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[1].set_title('Predicted U RMS')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlim(0, 1.0)
    axs[1].set_ylim(0, 1.0)
    axs[1].set_aspect('equal')

    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)

    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'rms_u_comparison.png'), dpi=600)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(size*width, size*width/2))
    c1 = axs[0].contourf(X, Y, rms_true_plot[1], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[0].set_title('True V RMS')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlim(0, 1.0)
    axs[0].set_ylim(0, 1.0)
    axs[0].set_aspect('equal')

    c2 = axs[1].contourf(X, Y, rms_pred_plot[1], levels=200, cmap='RdBu_r', vmin=0, vmax=1)
    axs[1].set_title('Predicted V RMS')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlim(0, 1.0)
    axs[1].set_ylim(0, 1.0)
    axs[1].set_aspect('equal')

    fig.colorbar(c1, ax=axs, shrink=0.8, ticks=ticks, format='%.2f', pad=0.03)

    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'rms_v_comparison.png') , dpi=600)
    plt.close()

    
        
    

def plot_tke(runner, true_path, pred_path, idx, eval_idx, true_idx):
    """
    Plot the TKE of the predicted and truth data over time.
    """
    time_lag = runner.config['params']['time_lag']

    with h5py.File(pred_path, 'r') as f:
        tke_pred = f['tke_pred'][:]

    with h5py.File(true_path, 'r') as f:
        tke_true = f[runner.paths_bib.latent_id + '_tke_true'][:]

    # TKE error
    tke_error = l2_err_norm(true=tke_true[eval_idx[time_lag:]], pred=tke_pred[time_lag:len(eval_idx)])

    print(f"TKE error: {100*tke_error:.3f}%")
    t = idx / 100
    t_true = np.array(true_idx) / 100

    size = 0.6
    plt.figure(figsize=(size*width,size*height))
    plt.plot(t_true, tke_true[true_idx], label='True TKE', color='k', linestyle='-')
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
    # plt.tight_layout()
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'tke_comparison.png'), dpi=600)
    plt.close()

    

    # psd of TKE
    f, Pxx_true = welch(tke_true[eval_idx[time_lag:]], fs=100)
    f, Pxx_pred = welch(tke_pred[time_lag:], fs=100)
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
    # plt.tight_layout() #rect=[0, 0, 1, 0.95]
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'tke_psd_comparison.png'), dpi=600)
    plt.close()

    


def plot_PSDs(runner, data_dict):
    psd_results = {}
    time_lag = runner.config['params']['time_lag']
    # Loop through data types (ground truth and prediction) and points
    for data_type, points in data_dict.items():
        psd_results[data_type] = {}
        for point_name, data in points.items():
            psd_results[data_type][point_name] = {}
            # Compute PSD for the u-component
            f_u, Pxx_u = psd(data[time_lag:, 0])
            f_v, Pxx_v = psd(data[time_lag:, 1])
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
    plt.tight_layout(rect=[0, 0, 1, 1.15])  # Reduce top margin for suptitle
    
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
    plt.tight_layout(rect=[0, 0, 1, 1.15])  # Reduce top margin for suptitle
    
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'psd_comparison_p2.png'), dpi=600)
    plt.close()

def plot_autocorr(runner, data_dict):
    """
    Plot the autocorrelation of the data. Not yet implemented
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
    # fig.tight_layout()
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'autocorr_comparison.png'), dpi=600)
    plt.close()



def plot_coherence(runner, data_dict, eval_idx, true_idx):
    """
    Plot the coherence between the predicted and truth data.
    """
    time_lag = runner.config['params']['time_lag']
    size = 1
    fig, axs = plt.subplots(1, 2, figsize=(size*width, size*width/3))
    if len(eval_idx) == len(true_idx):
        eval_idx = range(len(true_idx))
    # Compute Coherence for U and V component point 1
    truth = data_dict['truth']['p1'][eval_idx[time_lag:], :]
    pred = data_dict['pred']['p1'][time_lag:len(eval_idx), :]

    f_u, Cxy_u = coher(truth[:,0], pred[:,0])
    f_v, Cxy_v = coher(truth[:,1], pred[:,1])

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
    
    # fig.tight_layout(rect=[0, 0, 1, 1.1])  # Reduce top margin for suptitle
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'coherence_comparison_p1.png'), dpi=600)
    plt.close()

    # Compute Coherence for U and V component point 2

    truth = data_dict['truth']['p2'][eval_idx[time_lag:], :]
    pred = data_dict['pred']['p2'][time_lag:len(eval_idx), :]


    f_u, Cxy_u = coher(truth[:,0], pred[:,0])
    f_v, Cxy_v = coher(truth[:,1], pred[:,1])
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

    # fig.tight_layout(rect=[0, 0, 1, 1.1])  # Reduce top margin for suptitle
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
    # plt.tight_layout()

    plt.savefig(runner.paths_bib.fig_dir + 'points.png', dpi=600)
    plt.close()

def plot_point_data(runner, data_dict, idx, eval_idx, true_idx):
    """
    Plot the data at the points of interest.
    """
    time_lag = runner.config['params']['time_lag']
    size = 0.75
    fig, axs = plt.subplots(2, 2, figsize=(size*width, size*width/2), sharex=True, sharey=True)
    
    t = idx / 100  # Convert to nondimensional time
    t_true = np.array(true_idx) / 100

    if len(eval_idx) == len(true_idx):
        eval_idx = range(len(true_idx))

    # Plotting the data for Point 1
    axs[0,0].plot(t_true, data_dict['truth']['p1'][:,0], label='True', color='k', linestyle='-')
    axs[0,0].plot(t, data_dict['pred']['p1'][:,0], label='Predicted', color='r', linestyle='-.')
    axs[0,1].plot(t_true, data_dict['truth']['p1'][:,1], label='True', color='k', linestyle='-')
    axs[0,1].plot(t, data_dict['pred']['p1'][:,1], label='Predicted', color='r', linestyle='-.')
    
    # axs[0,0].set_title('Point 1 $u$-component')
    # axs[0,1].set_title('Point 1 $v$-component')
    
    axs[0,0].set_ylabel('$u_{p1}$')
    axs[0,1].set_ylabel('$v_{p1}$')
    
    # Plotting the data for Point 2
    axs[1,0].plot(t_true, data_dict['truth']['p2'][:,0], label='True', color='k', linestyle='-')
    axs[1,0].plot(t, data_dict['pred']['p2'][:,0], label='Predicted', color='r', linestyle='-.')
    axs[1,1].plot(t_true, data_dict['truth']['p2'][:,1], label='True', color='k', linestyle='-')
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

    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'point_data_comparison.png'), dpi=600)
    plt.close()

    # calculate L2 error for each point
    l2_error_p1 = l2_err_norm(true=data_dict['truth']['p1'][eval_idx[time_lag:]], pred=data_dict['pred']['p1'][time_lag:len(eval_idx)])
    l2_error_p2 = l2_err_norm(true=data_dict['truth']['p2'][eval_idx[time_lag:]], pred=data_dict['pred']['p2'][time_lag:len(eval_idx)])
    print(f"L2 error at Point 1: {100*l2_error_p1:.3f}%")
    print(f"L2 error at Point 2: {100*l2_error_p2:.3f}%")


def plot_spectrograms(runner, data_dict, idx, true_idx):
    """
    Plot the spectrograms of the point probes using scipy.signal.ShortTimeFFT.spectrogram.
    Compares ground truth and predictions for both u and v components, with separate plots for each point.
    """
    from scipy.signal import spectrogram
    size = 0.75
    fs = 100  # Sampling frequency
    time_lag = runner.config['params']['time_lag']
    fmt = lambda x, pos: r'$10^{{{}}}$'.format(int(np.log10(x))) if x != 0 else '0'

    for point in ['p1', 'p2']:
        fig, axs = plt.subplots(2, 2, figsize=(size*width*2, size*width), sharex=True, sharey=True)
        for i, comp in enumerate(['$u$', '$v$']):
            # 0: u, 1: v
            data_true = data_dict['truth'][point][:, i]
            data_pred = data_dict['pred'][point][time_lag:, i]

            # Compute spectrograms
            f_true, t_true, Sxx_true = spectrogram(data_true, fs=fs)
            f_pred, t_pred, Sxx_pred = spectrogram(data_pred, fs=fs)

            # # Plot ground truth
            # im0 = axs[i, 0].pcolormesh(t_true + true_idx[0]/100, f_true, 10 * np.log10(np.fmax(Sxx_true, 1e-12)), cmap='RdBu_r')
            # axs[i, 0].set_title(f'True {comp} at {point}')
            # axs[i, 0].set_ylabel('Nondimensional frequency')
            # fig.colorbar(im0, ax=axs[i, 0], format='%.1f', label='dB')

            # # Plot prediction
            # # print(f"Shape of f_pred: {f_pred.shape}, Sxx_pred: {Sxx_pred.shape}")
            # im1 = axs[i, 1].pcolormesh(t_pred + idx[time_lag]/100, f_pred, 10 * np.log10(np.fmax(Sxx_pred, 1e-12)), cmap='RdBu_r')
            # axs[i, 1].set_title(f'Predicted {comp} at {point}')
            # fig.colorbar(im1, ax=axs[i, 1], format='%.1f', label='dB')

            # Plot ground truth (no dB, log color scale)
            im0 = axs[i, 0].pcolormesh(
                t_true + true_idx[0]/100, f_true, Sxx_true,
                cmap='RdBu_r', shading='Gouraud',
                norm=matplotlib.colors.LogNorm(vmin=np.fmax(Sxx_true, 1e-12).min(), vmax=Sxx_true.max())
            )
            axs[i, 0].set_title(f'True {comp} at {point}')
            axs[i, 0].set_ylabel('Nondimensional frequency')
            cbar0 = fig.colorbar(im0, ax=axs[i, 0], format=FuncFormatter(fmt), 
                                 label='Power Spectral Density', shrink=0.8, pad=0.03)

            # Plot prediction (no dB, log color scale)
            im1 = axs[i, 1].pcolormesh(
                t_pred + idx[time_lag]/100, f_pred, Sxx_pred,
                cmap='RdBu_r', shading='Gouraud',
                norm=matplotlib.colors.LogNorm(vmin=np.fmax(Sxx_pred, 1e-12).min(), vmax=Sxx_pred.max())
            )
            axs[i, 1].set_title(f'Predicted {comp} at {point}')
            cbar1 = fig.colorbar(im1, ax=axs[i, 1], format=FuncFormatter(fmt), label='Power Spectral Density', shrink=0.8, pad=0.03)
            

        for ax in axs[-1, :]:
            ax.set_xlabel('Nondimensional time')
        # plt.tight_layout()
        plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, f'spectrogram_{point}.png'), dpi=600)
        plt.close()

def plot_phase_portraits(runner, data_dict):
    """
    Plot phase portraits (u vs v) for points 1 and 2, comparing predictions and ground truth.
    """
    time_lag = runner.config['params']['time_lag']
    size = 0.6
    for point in ['p1', 'p2']:
        plt.figure(figsize=(size*width, size*width))
        # Ground truth
        plt.plot(
            data_dict['truth'][point][time_lag:, 0],
            data_dict['truth'][point][time_lag:, 1],
            label='True',
            color='k',
            linestyle='--'
        )
        # Prediction
        plt.plot(
            data_dict['pred'][point][time_lag:, 0],
            data_dict['pred'][point][time_lag:, 1],
            label='Predicted',
            color='r',
            linestyle='-.'
        )
        plt.xlabel(f'$u_{{{point}}}$')
        plt.ylabel(f'$v_{{{point}}}$')
        plt.title(f'Phase Portrait at {point.upper()}', pad=16)
        plt.legend(
            ['True', 'Predicted'],
            loc='lower right',
            bbox_to_anchor=(1.03, 0.985),  # Adjust position to the right of the plot
            ncol=2,  # Spread horizontally
            frameon=False,  # Removes legend border,
            fontsize=8  # Adjust font size
        )
        plt.grid(visible=True, linestyle='--', linewidth=0.5)
        # plt.tight_layout()
        plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, f'phase_portrait_{point}.png'), dpi=600)
        plt.close()



def attention_maps(runner):
    """
    Plot the attention maps for the predicted data.
    """
    
    time_lag = runner.config['params']['time_lag']
    # get val_indices
    val_indices = runner.val_indices[:time_lag]
    # load dof_u and dof_v
    with h5py.File(runner.paths_bib.latent_path, 'r') as f:
        if runner.config['latent_type'] == 'dls':
            dof_u = f['dof_u'][val_indices]
            dof_v = f['dof_v'][val_indices]

            initial_input = np.concatenate((dof_u, dof_v), axis=1)
        elif runner.config['latent_type'] == 'pod':
            initial_input = f['dofs'][val_indices, :runner.config['latent_params']['num_modes']]

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
                    

        # plt.tight_layout()
        plt.savefig(runner.paths_bib.fig_dir + 'attention_weights.png', dpi=600)
        plt.close()


def coeff_PDF(runner, data_dict, eval_idx, true_idx):
    """
    Plot the probability density function of the predicted and truth data.
    """
    time_lag = runner.config['params']['time_lag']
    size = 0.75
    fig, axs = plt.subplots(2, 2, figsize=(size*width, size*width/2))

    # Coefficients for Point 1
    coeff_true_p1 = data_dict['truth']['p1'][time_lag:, :]
    coeff_pred_p1 = data_dict['pred']['p1'][time_lag:, :]

    axs[0, 0].hist(coeff_true_p1[:, 0], bins=50, density=True, alpha=0.5, label='True $u_{p1}$', color='k')
    axs[0, 0].hist(coeff_pred_p1[:, 0], bins=50, density=True, alpha=0.5, label='Predicted $u_{p1}$', color='r')
    axs[0, 0].set_ylabel('PDF($u_{p1}$)')
    # axs[0, 0].set_title('PDF of Coefficients at Point 1 (u-component)')
    
    axs[0, 1].hist(coeff_true_p1[:, 1], bins=50, density=True, alpha=0.5, label='True $v_{p1}$', color='k')
    axs[0, 1].hist(coeff_pred_p1[:, 1], bins=50, density=True, alpha=0.5, label='Predicted $v_{p1}$', color='r')
    axs[0, 1].set_ylabel('PDF($v_{p1}$)')
    # axs[0, 1].set_title('PDF of Coefficients at Point 1 (v-component)')

    # Coefficients for Point 2
    coeff_true_p2 = data_dict['truth']['p2'][time_lag:, :]
    coeff_pred_p2 = data_dict['pred']['p2'][time_lag:, :]

    axs[1, 0].hist(coeff_true_p2[:, 0], bins=50, density=True, alpha=0.5, label='True $u_{p2}$', color='k')
    axs[1, 0].hist(coeff_pred_p2[:, 0], bins=50, density=True, alpha=0.5, label='Predicted $u_{p2}$', color='r')
    # axs[1, 0].set_title('PDF of Coefficients at Point 2 (u-component)') 
    axs[1, 1].hist(coeff_true_p2[:, 1], bins=50, density=True, alpha=0.5, label='True $v_{p2}$', color='k')
    axs[1, 1].hist(coeff_pred_p2[:, 1], bins=50, density=True, alpha=0.5, label='Predicted $v_{p2}$', color='r')
    # axs[1, 1].set_title('PDF of Coefficients at Point 2 (v-component)')
    axs[1, 0].set_ylabel('PDF($u_{p2}$)')
    axs[1, 1].set_ylabel('PDF($v_{p2}$)')
    for ax in axs.flat:
        ax.grid(visible=True, linestyle='--', linewidth=0.5)    
    axs[0,1].legend(
        ['True', 'Predicted'],
        loc='lower right',
        bbox_to_anchor=(1.05, 0.9),  # Adjust position to the right of the plot
        ncol=2,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8  # Adjust font size
    )
    fig.suptitle('Probability Density Function of Velocity at Points of Interest')
    fig.tight_layout(rect=[0, 0, 1, 1.1])
    plt.savefig(os.path.join(runner.paths_bib.pred_fig_dir, 'coeff_pdf_comparison.png'), dpi=600)
    plt.close()


def animate(runner):
    """
    Create an animation of the truth and predicted data.
    """
    import imageio


    plt.rcParams['font.size'] = 20 # Change default font size to 12
    plt.rcParams['axes.titlesize'] = 28 # Change axes title font size
    plt.rcParams['axes.labelsize'] = 20 # Change axes labels font size
    plt.rcParams['xtick.labelsize'] = 20 # Change x-axis tick labels font size
    plt.rcParams['ytick.labelsize'] = 20 # Change y-axis tick labels font size
    plt.rcParams['legend.fontsize'] = 20 # Change legend font size

    time_lag = runner.config['params']['time_lag']
    time_lim = 100000
    run_lim = 30 # 60 second max animation length
    frame_rate = 30 # Frames per second
    nx = runner.l_config.nx
    ny = runner.l_config.ny
    nx_t = runner.l_config.nx_t
    ny_t = runner.l_config.ny_t
    # t = idx / 100  # Convert to nondimensional time

    for pred_file in os.listdir(runner.paths_bib.predictions_dir):
        if pred_file.endswith('.h5') and 'rec' in pred_file and any(key in pred_file for key in runner.config['predictions'].keys()):
            print(f"Loading predictions from {pred_file}")
            pred_name = pred_file.replace('rec_', '').replace('_pred.h5', '')
            with h5py.File(os.path.join(runner.paths_bib.predictions_dir, pred_file), 'r') as f:
                num_snaps = f['Q_rec'].shape[0]
                if num_snaps > time_lim:
                    num_snaps = time_lim

                # Pick frame_skip such that max length is less than run_lim seconds
                frame_skip = int(max(1, np.ceil(num_snaps / (run_lim * frame_rate))))
                print(f"Number of snapshots: {num_snaps}, Frame skip: {frame_skip}")
                print(f"Vid_length: {(num_snaps // frame_skip )/ frame_rate} seconds")

                pred = f['Q_rec'][time_lag:num_snaps:frame_skip]
                idx = f['idx'][time_lag:num_snaps:frame_skip]
                tke_pred = f['tke_pred'][:num_snaps]

            print(f'Loading truth from {runner.paths_bib.data_path}')
            with h5py.File(runner.paths_bib.data_path, 'r') as f:
                true_num_snaps = f['UV'].shape[0]
                mean = f['mean'][:nx_t, :ny_t]
                truth = np.zeros(pred.shape, dtype=np.float32)
                for i in range(len(idx)):
                    truth[i] = f['UV'][min(idx[i],true_num_snaps-1), :nx_t, :ny_t, :] - mean
                tke_true = f[runner.paths_bib.latent_id + '_tke_true'][idx[0]-time_lag:idx[0]-time_lag+num_snaps]
            
            # Root mean squared error normalized by range of truth
            error_plot = 100 * abs(truth - pred) / np.max(np.abs(truth), axis=(0,1,2), keepdims=True)

            Q_plot = truth / np.max(np.abs(truth), axis=(0,1,2), keepdims=True)
            Q_plot_pred = pred / np.max(np.abs(truth), axis=(0,1,2), keepdims=True)

            num_overlap = (true_num_snaps - idx[0]) // frame_skip
            frames = []
            for i, id in enumerate(idx):  # Number of frames
                if i % 100 == 0:
                    print(f'Processing frame {i+1}/{len(idx)}')
                fig, axs = plt.subplots(3, 3, figsize=(16, 15), gridspec_kw={'height_ratios': [1, 1, 0.6]})

                # Top two rows: velocity and error plots
                axs[0, 0].imshow(Q_plot[i, :, :, 0], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[0, 0].set_title('True $u$')
                axs[1, 0].imshow(Q_plot[i, :, :, 1], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[1, 0].set_title('True $v$')
                axs[0, 1].imshow(Q_plot_pred[i, :, :, 0], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[0, 1].set_title('Predicted $u$')
                axs[1, 1].imshow(Q_plot_pred[i, :, :, 1], cmap='seismic', origin='lower', vmin=-1, vmax=1)
                axs[1, 1].set_title('Predicted $v$')

                im = axs[0, 2].imshow(error_plot[min(i, num_overlap), :, :, 0], cmap='RdBu_r', origin='lower', vmin=0, vmax=100)
                axs[0, 2].set_title('Error\% $u$')
                axs[0, 2].set_aspect('equal')
                divider = make_axes_locatable(axs[0, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, format='%.0f')
                im.set_clim(vmin=0, vmax=100)
                cbar.set_label('Error (\%)', rotation=270, labelpad=15)

                im = axs[1, 2].imshow(error_plot[min(i, num_overlap), :, :, 1], cmap='RdBu_r', origin='lower', vmin=0, vmax=100)
                axs[1, 2].set_title('Error\% $v$')
                axs[1, 2].set_aspect('equal')
                divider = make_axes_locatable(axs[1, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, format='%.0f')
                im.set_clim(vmin=0, vmax=100)
                cbar.set_label('Error (\%)', rotation=270, labelpad=15)

                for ax in axs[:2, :].flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)

                t = (np.arange(idx[0], idx[0] + num_snaps)-time_lag) / 100  
                t_true = t[:len(tke_true)]
                # Bottom row: TKE plot spanning all columns
                ax_tke = plt.subplot2grid((3, 3), (2, 0), colspan=3, fig=fig)
                ax_tke.plot(t_true, tke_true, label='True TKE', color='k')
                ax_tke.plot(t, tke_pred, label='Predicted TKE', color='r', linestyle='--')
                ax_tke.axvline(id/100, color='b', linestyle=':', linewidth=2, label='Current Frame')
                ax_tke.set_ylabel('TKE')
                ax_tke.set_xlabel('Nondimensional time')
                ax_tke.legend(
                    ['True', 'Predicted'],
                    loc='lower right',
                    bbox_to_anchor=(1, 0.95),  # Adjust position to the right of the plot
                    ncol=2,  # Spread horizontally
                    frameon=False,  # Removes legend border,
                    fontsize=20  # Adjust font size
                )
                ax_tke.set_title('True vs Predicted TKE')
                ax_tke.grid(visible=True, linestyle='--', linewidth=0.5)

                # plt.tight_layout()

                # Save the current frame
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)

            # Save as GIF
            # imageio.mimsave(runner.paths_bib.anim_dir + pred_name + '.gif', frames, fps=30, loop=0)

            # save as mp4
            writer = imageio.get_writer(
                runner.paths_bib.anim_dir + pred_name + '.mp4',
                fps=frame_rate,
                codec='libx264',  # Use H.264 codec for better quality control
                quality=8,        # 0 (lowest) to 10 (highest), default is 5
                ffmpeg_params=['-crf', '18']  # Lower CRF = higher quality (default 23, range 0–51)
            )
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
