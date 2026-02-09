import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os
import numpy as np
import h5py
import torch
from scipy.signal import welch, correlate, coherence, correlation_lags, csd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import math
from lib.plotting import curl


paper_width = 470 # pt
width = paper_width / 72.27 # inches
height = width / 1.618 # inches
plt.rcParams['text.usetex'] = True
# change font to helvetica
# plt.rcParams['font.family'] = 'Helvetica'
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



def compare_RMS(comp_config):
    """
    Compare the RMS of multiple models against the true data in a subplot for each model for publication. 
    """
    
    comparisons = comp_config['comparisons']
    n_models = len(comparisons)

    RMS_max = np.max(comparisons[1]['results']['rms_true'], axis=(1,2), keepdims=True)
    rms_true_plot = comparisons[1]['results']['rms_true'] / RMS_max

    X = comparisons[1]['results']['X_grid']
    Y = comparisons[1]['results']['Y_grid']

    fig, axes = plt.subplots(2, 1+n_models, figsize=(width, 0.8*height), dpi=600, sharex=True, sharey=True)

    im = axes[0, 0].contourf(X, Y, rms_true_plot[0], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
    axes[1, 0].contourf(X, Y, rms_true_plot[1], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth')
    axes[1, 0].set_xlabel('L2 Error = ')

    for i, case in enumerate(comparisons):
        rms_pred_plot = case['results']['rms_pred'] / RMS_max
        error = l2_err_norm(case['results']['rms_true'], case['results']['rms_pred'])

        X = comparisons[i]['results']['X_grid']
        Y = comparisons[i]['results']['Y_grid']

        axes[0, i+1].contourf(X, Y, rms_pred_plot[0], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
        axes[1, i+1].contourf(X, Y, rms_pred_plot[1], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
        axes[0, i+1].set_title(case['label'])
        axes[1, i+1].set_xlabel(f'{100*error:.3f}\%')


    for ax in axes.flatten():
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.label_outer()
        ax.set_aspect('equal', 'box')

    axes[0,0].set_ylabel('$\mathrm{RMS} \, u$')
    axes[1,0].set_ylabel('$\mathrm{RMS} \, v$')

    ticks = np.linspace(0, 1, 6)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01, cmap='RdBu_r', format='%.2f', ticks=ticks)
    cbar.set_label('Normalized RMS', rotation=270, labelpad=15)
    plt.savefig(os.path.join(comp_config['save_dir'], 'RMS_comparison.png'), dpi=600, bbox_inches='tight')
        

def compare_tke(comp_config):
    """
    Compare the TKE of multiple models against the true data in a subplot for each model for publication. 
    """
    
    comparisons = comp_config['comparisons']
    n_models = len(comparisons)

    
    tke_true_plot = comparisons[1]['results']['tke_true']

    true_idx = comparisons[1]['results'].get('true_idx', 0)

    size = 0.75
    plt.figure(figsize=(size*width, size*height), dpi=600)
    plt.plot(np.array(true_idx) / 100, tke_true_plot[true_idx], '-', label='Ground Truth', color='gray', linewidth=1.5)
    for i, case in enumerate(comparisons):
        
        tke_pred_plot = case['results']['tke_pred']
        
        pred_idx = case['results'].get('pred_idx', 0)
        plt.plot(np.array(pred_idx) / 100, tke_pred_plot, linestyle=case['line_style'], label=case['label'], color=case.get('color', None), linewidth=1)

    plt.xlabel('Nondimensional time')
    plt.ylabel(r'$\mathrm{TKE} = \frac{1}{2} \sum \mathbf{u}^2$')
    # plt.title('Comparison of True and Predicted TKE', pad=16)
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1.025, 0.975),  # Adjust position to the right of the plot
        ncol=n_models+1,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8,  # Adjust font size
        columnspacing=1.0
    )
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    plt.savefig(os.path.join(comp_config['save_dir'], 'TKE_comparison.png'), dpi=600)
    plt.close()


    # frequency spectrum
    plt.figure(figsize=(size*width, size*height), dpi=600)
    f, Pxx_true = welch(comparisons[1]['results']['tke_true'], fs=100, nperseg=256)
    plt.loglog(f, Pxx_true, '-', label='Ground Truth', color='gray', linewidth=1.5)
    for i, case in enumerate(comparisons):      
        f, Pxx_pred = welch(case['results']['tke_pred'], fs=100, nperseg=256)
        plt.loglog(f, Pxx_pred, linestyle=case['line_style'], label=case['label'], color=case.get('color', None), linewidth=1) 
    plt.xlabel('Nondimensional Frequency')
    plt.ylabel('PSD($\mathrm{TKE}$)')
    # plt.title('Power Spectral Density of TKE', pad=16)
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1.025, 0.975),  # Adjust position to the right of the plot
        ncol=n_models+1,  # Spread horizontally
        frameon=False,  # Removes legend border,
        fontsize=8,  # Adjust font size
        columnspacing=1.0
    )
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(comp_config['save_dir'], 'TKE_PSD_comparison.png'), dpi=600, bbox_inches='tight')
    plt.close()


    # Create a subplot where each TKE is on a different subplot and all PSDs share one axis
    size = 1
    fig, axes = plt.subplots(2, math.ceil((1+n_models)/2), figsize=(size*width, size*height), dpi=600)
    axes = axes.flatten()

    axes[n_models].loglog(f, Pxx_true, '-', label='Ground Truth', color='gray', linewidth=1.5)
    axes[n_models].set_xlabel('Nondimensional Frequency')
    axes[n_models].set_ylabel('PSD($\mathrm{TKE}$)')
    axes[n_models].set_title('Power Spectral Density of TKE')


    for i, case in enumerate(comparisons):
        tke_true_plot = case['results']['tke_true']
        true_idx = case['results'].get('true_idx', 0)
        tke_pred_plot = case['results']['tke_pred']
        pred_idx = case['results'].get('pred_idx', 0)
        axes[i].plot(np.array(true_idx) / 100, tke_true_plot[true_idx], '-', label='Ground Truth',color='gray', linewidth=1.5)
        axes[i].plot(np.array(pred_idx) / 100, tke_pred_plot, linestyle=case['line_style'], label=case['label'], color=case.get('color', None), linewidth=1)
        axes[i].set_title(case['label'])
        axes[i].set_xlabel('Nondimensional time')
        axes[i].set_ylabel(r'$\mathrm{TKE} = \frac{1}{2} \sum \mathbf{u}^2$')

        # axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axes[i].grid(visible=True, linestyle='--', linewidth=0.5)
        
        f, Pxx_pred = welch(case['results']['tke_pred'], fs=100, nperseg=256)
        axes[n_models].loglog(f, Pxx_pred, linestyle=case['line_style'], label=case['label'], color=case.get('color', None), linewidth=1)

    
    

    # Collect handles and lables from one of the axes
    handles_labels = [axes[n_models].get_legend_handles_labels()]
    handles, labels = [], []
    for hl in handles_labels:
        for handle, label in zip(*hl):
            if label not in labels:  # Avoid duplicate labels
                labels.append(label)
                handles.append(handle)

    # Create a figure-level legend
    fig.legend(
        handles, labels,
        loc='lower right',
        bbox_to_anchor=(1, 0),
        ncol=n_models+1,
        frameon=False,
        fontsize=8,
        columnspacing=1.0
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(comp_config['save_dir'], 'TKE_comparison_subplots.png'), dpi=600, bbox_inches='tight')
    plt.close()

    



def compare_pdf(comp_config):
    """
    Compare the PDF of multiple models against the true data in a subplot for each model for publication. 
    """
    from seaborn import kdeplot
    
    comparisons = comp_config['comparisons']
    n_models = len(comparisons)
    time_lag = comparisons[1]['results'].get('time_lag', 0)
    truth = comparisons[1]['results']['point_dict']['truth']


    size = 0.8
    fig, axs = plt.subplots(2, 2, figsize=(size*width, size*height))
    # Coefficients for Point 1
    coeff_true_p1 = truth['p1'][time_lag:, :]
    kdeplot(coeff_true_p1[:, 0], ax=axs[0, 0], label='Ground Truth', color='gray', common_norm=True, linewidth=1.5)
    axs[0, 0].set_title('PDF($u_{p1}$)')

    kdeplot(coeff_true_p1[:, 1], ax=axs[0, 1], label='True $v_{p1}$', color='gray', common_norm=True, linewidth=1.5)
    axs[0, 1].set_title('PDF($v_{p1}$)')


    # Coefficients for Point 2
    coeff_true_p2 = truth['p2'][time_lag:, :]
    kdeplot(coeff_true_p2[:, 0], ax=axs[1, 0], label='True $u_{p2}$', color='gray', common_norm=True, linewidth=1.5)
    axs[1, 0].set_title('PDF($u_{p2}$)')

    kdeplot(coeff_true_p2[:, 1], ax=axs[1, 1], label='True $v_{p2}$', color='gray', common_norm=True, linewidth=1.5)
    axs[1, 1].set_title('PDF($v_{p2}$)')

    # Now plot predictions from each model
    for i, case in enumerate(comparisons):
        pred = case['results']['point_dict']['pred']
        time_lag = case['results'].get('time_lag', 0)

        coeff_pred_p1 = pred['p1'][time_lag:, :]
        kdeplot(coeff_pred_p1[:, 0], ax=axs[0, 0], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True, linewidth=1)
        kdeplot(coeff_pred_p1[:, 1], ax=axs[0, 1], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True, linewidth=1)

        coeff_pred_p2 = pred['p2'][time_lag:, :]
        kdeplot(coeff_pred_p2[:, 0], ax=axs[1, 0], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True, linewidth=1)
        kdeplot(coeff_pred_p2[:, 1], ax=axs[1, 1], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True, linewidth=1)


    for ax in axs.flat:
        # ax.grid(visible=True, linestyle='--', linewidth=0.5)  
        ax.set_ylabel('')  
        ax.set_yticks([])
    # Collect handles and lables from one of the axes
    handles_labels = [axs[0,0].get_legend_handles_labels()]
    handles, labels = [], []
    for hl in handles_labels:
        for handle, label in zip(*hl):
            if label not in labels:  # Avoid duplicate labels
                labels.append(label)
                handles.append(handle)

    # Create a figure-level legend
    fig.legend(
        handles, labels,
        loc='lower right',
        bbox_to_anchor=(1, 0),
        ncol=n_models+1,
        frameon=False,
        fontsize=8,
        columnspacing=1.0
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    # fig.suptitle('Probability Density Function of Velocity at Points of Interest')
    plt.savefig(os.path.join(comp_config['save_dir'], 'PDF_comparison.png'), dpi=600, bbox_inches='tight')
    plt.close()


def compare_phase_portrait(comp_config):
    """
    Compare the phase portrait of multiple models against the true data in a subplot for each model for publication. 
    """
    comparisons = comp_config['comparisons']
    n_models = len(comparisons)
    time_lag = comparisons[1]['results'].get('time_lag', 0)
    truth = comparisons[1]['results']['point_dict']['truth']
    size = 1
    fig, axs = plt.subplots(2, n_models, figsize=(size*width, 1.2*size*height), sharex=False, sharey=True)

    coeff_true_p1 = truth['p1'][time_lag:, :]
    coeff_true_p2 = truth['p2'][time_lag:, :]


    # Now plot predictions from each model
    for i, case in enumerate(comparisons):  
        pred = case['results']['point_dict']['pred']
        time_lag = case['results'].get('time_lag', 0)

        coeff_pred_p1 = pred['p1'][time_lag:, :]
        axs[0, i].plot(coeff_true_p1[:, 0], coeff_true_p1[:, 1], label='Ground Truth', color='gray', linestyle='--', linewidth=1.5)
        axs[0, i].plot(coeff_pred_p1[:, 0], coeff_pred_p1[:, 1], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), linewidth=1.5)
        # axs[0, i+1].set_title(f"Phase Portrait: {case['label']}")
        axs[0, i].set_xlabel('$u_{p1}$')
        axs[0, i].set_ylabel('$v_{p1}$')

        coeff_pred_p2 = pred['p2'][time_lag:, :]
        axs[1, i].plot(coeff_true_p2[:, 0], coeff_true_p2[:, 1], label='Ground Truth', color='gray', linestyle='--', linewidth=1.5)
        axs[1, i].plot(coeff_pred_p2[:, 0], coeff_pred_p2[:, 1], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), linewidth=1.5)
        # axs[1, i+1].set_title(f"Phase Portrait: {case['label']}")
        axs[1, i].set_xlabel('$u_{p2}$')
        axs[1, i].set_ylabel('$v_{p2}$')

        axs[0, i].set_title(f"{case['label']}")


    # Collect all handles and labels from all axes
    handles_labels = []
    for ax in axs.flatten():
        handles_labels.append(ax.get_legend_handles_labels())
    handles, labels = [], []
    for hl in handles_labels:
        for handle, label in zip(*hl):
            if label not in labels:  # Avoid duplicate labels
                labels.append(label)
                handles.append(handle)
    # Create a figure-level legend
    fig.legend(
        handles, labels,
        loc='lower right',
        bbox_to_anchor=(1, 0),
        ncol=n_models+1,
        frameon=False,
        fontsize=8,
        columnspacing=1.0
    )

    for ax in axs.flatten():
        ax.grid(visible=True, linestyle='--', linewidth=0.5)

    # xticks_p1 = np.round(np.linspace(np.min(coeff_true_p1[:,0]), np.max(coeff_true_p1[:,0]), 5), 2)
    # xticks_p2 = np.round(np.linspace(np.min(coeff_true_p2[:,0]), np.max(coeff_true_p2[:,0]), 5), 2)
    # yticks = np.round(np.linspace(np.min(np.concatenate([coeff_true_p1[:,1],coeff_true_p2[:,1]],axis=0)), np.max(np.concatenate([coeff_true_p1[:,1],coeff_true_p1[:,1]],axis=0)), 5), 2)

    # for ax in axs[0, :]:
    #     ax.set_xticks(xticks_p1[1:-1])
    #     ax.set_yticks(yticks[1:-1])
    # for ax in axs[1, :]:
    #     ax.set_xticks(xticks_p2[1:-1])
    #     ax.set_yticks(yticks[1:-1])

    if "30k" in comp_config.get("save_dir"):
        xticks_p1 = 3*np.array([-0.02, 0, 0.02, 0.04])
        xticks_p2 = 5*np.array([-0.02, -0.01, 0, 0.01])
        yticks_p1 = 3*np.array([-0.04, -0.03, -0.02, -0.01, 0, 0.01])
        yticks_p2 = 3*np.array([-0.04, -0.03, -0.02, -0.01, 0, 0.01])

        for ax in axs[0, :]:
            ax.set_xticks(xticks_p1)
            ax.set_yticks(yticks_p1)
        for ax in axs[1, :]:
            ax.set_xticks(xticks_p2)
            ax.set_yticks(yticks_p2)

    # xticks_p2 = np.round(np.linspace(np.min(coeff_true_p2[:,0]), np.max(coeff_true_p2[:,0]), 5), 2)
    # yticks = np.round(np.linspace(np.min(np.concatenate([coeff_true_p1[:,1],coeff_true_p2[:,1]],axis=0)), np.max(np.concatenate([coeff_true_p1[:,1],coeff_true_p1[:,1]],axis=0)), 5), 2)

    # for ax in axs[0, :]:
    #     ax.set_xticks(xticks_p1[1:-1])
    #     ax.set_yticks(yticks[1:-1])
    # for ax in axs[1, :]:
    #     ax.set_xticks(xticks_p2[1:-1])
    #     ax.set_yticks(yticks[1:-1])

    # # make axlim = 1.2 * max abs data value for each subplot
    # for i in range(axs.shape[0]):
    #     for j in range(axs.shape[1]):
    #         ax = axs[i, j]
    #         all_x_data = []
    #         all_y_data = []
    #         for line in ax.get_lines():
    #             all_x_data.extend(line.get_xdata())
    #             all_y_data.extend(line.get_ydata())
    #         max_val_x = np.max(np.abs(all_x_data))
    #         max_val_y = np.max(np.abs(all_y_data))
    #         xlim = 1.2 * max_val_x
    #         ylim = 1.2 * max_val_y
    #         ax.set_xlim(-xlim, xlim)
    #         ax.set_ylim(-ylim, ylim)

    # double font size for all figure text elements and legend
    # for ax in axs.flatten():
    #     for item in ([ax.xaxis.label, ax.yaxis.label] +
    #                  ax.get_xticklabels() + ax.get_yticklabels()):
    #         item.set_fontsize(16)

    #     ax.title.set_fontsize(24)


    

    

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    # fig.suptitle('Phase Portraits of Velocity at Points of Interest')
    plt.savefig(os.path.join(comp_config['save_dir'], 'Phase_Portrait_comparison.png'), dpi=600, bbox_inches='tight')
    plt.close()


def compare_anim(comp_config):
    """
    Create an animation comparing the vorticity fields of model predictions against the true data.
    Display the turbulent kinetic energy (TKE) below the vorticity contours, with all models and ground truth shown together for each time step.
    """

    import imageio
    
    comparisons = comp_config['comparisons']
    n_models = len(comparisons)
    run_lim = 30
    frame_rate = 30
    
    grid_x_true = comparisons[1]['results']['X_grid']
    grid_y_true = comparisons[1]['results']['Y_grid']

    truth_path = comparisons[1]['results']['true_path']
    
    # Pre-load all data before animation loop
    with h5py.File(truth_path, 'r') as f:
        vort_true_full = f[comparisons[1]['results']['paths_bib'].latent_id + '_vort_true'][:]
    
    tke_true = comparisons[1]['results']['tke_true']
    true_idx = comparisons[1]['results'].get('true_idx', np.arange(len(tke_true)))
    num_time_steps = comparisons[1]['results']['pred_idx'].shape[0]
    snap_lim = 2500
    num_time_steps = min(num_time_steps, snap_lim)
    total_snaps = len(tke_true)
    time_lag = comparisons[1]['results'].get('time_lag', 0)
    frame_skip = int(max(1, np.ceil(num_time_steps / (run_lim * frame_rate))))
    
    print(f"Number of snapshots: {num_time_steps}, Frame skip: {frame_skip}")
    print(f"Vid_length: {(num_time_steps // frame_skip) / frame_rate} seconds")
    
    plot_idx = comparisons[1]['results']['pred_idx'][time_lag:num_time_steps:frame_skip]
    vort_true = vort_true_full[plot_idx] if plot_idx[-1] < total_snaps else vort_true_full[:total_snaps:frame_skip]
    vmax = np.max(np.abs(vort_true))

    # Pre-load all prediction data
    vort_pred = []
    pred_data = []
    for i, case in enumerate(comparisons):
        pred_path = case['results']['pred_path']
        with h5py.File(pred_path, 'r') as f_pred:
            pred_idx = case['results']['pred_idx'][case['results'].get('time_lag', 0):snap_lim-1:frame_skip]
            vort_pred.append(f_pred['vort_pred'][pred_idx][:])
            
        pred_data.append({
            'grid_x': case['results']['X_grid'],
            'grid_y': case['results']['Y_grid'],
            'tke_pred': case['results']['tke_pred'],
            'pred_idx': case['results']['pred_idx'],
            'time_lag': case['results'].get('time_lag', 0),
            'line_style': case['line_style'],
            'label': case['label'],
            'color': case.get('color', None)
        })

    # Pre-create figure once
    size = 1
    fig, axs = plt.subplots(1, n_models+1, figsize=(size*width, 0.5*size*height))

    # fig, axs = plt.subplots(2, n_models+1, figsize=(size*width, size*height), dpi=100)
    # ax_tke = plt.subplot2grid((2, n_models+1), (1, 0), colspan=4, fig=fig)
    # plt.subplots_adjust(bottom=0.15, right=0.95)
    
    
    frames = []
    for t, id in enumerate(plot_idx):
        # Clear only data, not redraw the entire figure
        for ax in axs[:]:
            ax.clear()
        
        # Plot ground truth
        axs[0].contourf(grid_x_true, grid_y_true, vort_true[t], levels=100, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axs[0].set_title('Ground Truth')
        axs[0].set_aspect('equal', 'box')

        # Plot each model
        for i in range(n_models):
            axs[i+1].contourf(pred_data[i]['grid_x'], pred_data[i]['grid_y'], vort_pred[i][t], 
                                 levels=100, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axs[i+1].set_title(pred_data[i]['label'])
            axs[i+1].set_aspect('equal', 'box')
        
        for ax in axs[:]:
            ax.set_xticks([])
            ax.set_yticks([])

        # # Plot TKE signal
        # m = np.where(true_idx <= id)[0]
        # if len(m) > 0:
        #     ax_tke.plot(np.array(true_idx[m[0]:m[-1]]) / 100, tke_true[m[0]:m[-1]], 
        #                '-', label='Ground Truth', color='gray', linewidth=1.5)
        
        # for i, case in enumerate(pred_data):
        #     m = np.where(case['pred_idx'] <= id)[0]
        #     if len(m) > 0:
        #         ax_tke.plot(np.array(case['pred_idx'][m[0]:m[-1]]) / 100, case['tke_pred'][m[0]:m[-1]], 
        #                    linestyle=case['line_style'], label=case['label'], color=case['color'], linewidth=1)
        
        # ax_tke.set_xlabel('Nondimensional time')
        # ax_tke.set_ylabel(r'$\mathrm{TKE} = \frac{1}{2} \sum \mathbf{u}^2$')
        # ax_tke.grid(visible=True, linestyle='--', linewidth=0.5)
        # if t == 0:
        #     print("Adding legend to TKE plot")
        #     fig.legend(loc='lower right', bbox_to_anchor=(0.98, 0), ncol=n_models+1, 
        #        frameon=False, fontsize=8, columnspacing=1.0)

        # Capture frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        if (t+1) % frame_rate == 0:
            print(f"Processed frame {t+1} / {len(plot_idx)}")
    
    plt.close(fig)
    
    # Save as mp4
    writer = imageio.get_writer(os.path.join(comp_config['save_dir'], 'vort_comparison.mp4'),
                               fps=frame_rate, codec='libx264', quality=10, ffmpeg_params=['-crf', '15'])
    for frame in frames:
        writer.append_data(frame)
    writer.close()


    