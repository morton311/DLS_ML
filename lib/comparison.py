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


paper_width = 470 # pt
width = paper_width / 72.27 # inches
height = width / 1.618 # inches
plt.rcParams['text.usetex'] = True
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

    fig, axes = plt.subplots(2, 1+n_models, figsize=(width, 0.75*height), dpi=600, sharex=True, sharey=True)

    im = axes[0, 0].contourf(X, Y, rms_true_plot[0], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
    axes[1, 0].contourf(X, Y, rms_true_plot[1], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth')

    for i, case in enumerate(comparisons):
        rms_pred_plot = case['results']['rms_pred'] / RMS_max

        X = comparisons[i]['results']['X_grid']
        Y = comparisons[i]['results']['Y_grid']

        axes[0, i+1].contourf(X, Y, rms_pred_plot[0], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
        axes[1, i+1].contourf(X, Y, rms_pred_plot[1], levels = 200, cmap='RdBu_r', vmin=0, vmax=1)
        axes[0, i+1].set_title(case['label'])


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
    plt.savefig(os.path.join(comp_config['save_dir'], 'RMS_comparison.png'), dpi=600)
        

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
    plt.plot(np.array(true_idx) / 100, tke_true_plot, 'k-', label='Ground Truth')
    for i, case in enumerate(comparisons):
        
        tke_pred_plot = case['results']['tke_pred']
        
        pred_idx = case['results'].get('pred_idx', 0)
        plt.plot(np.array(pred_idx) / 100, tke_pred_plot, linestyle=case['line_style'], label=case['label'], color=case.get('color', None))

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
    plt.loglog(f, Pxx_true, 'k-', label='Ground Truth')
    for i, case in enumerate(comparisons):      
        f, Pxx_pred = welch(case['results']['tke_pred'], fs=100, nperseg=256)
        plt.loglog(f, Pxx_pred, linestyle=case['line_style'], label=case['label'], color=case.get('color', None)) 
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
    plt.savefig(os.path.join(comp_config['save_dir'], 'TKE_PSD_comparison.png'), dpi=600)
    plt.close()


    # Create a subplot where each TKE is on a different subplot and all PSDs share one axis
    size = 1
    fig, axes = plt.subplots(2, math.ceil((1+n_models)/2), figsize=(size*width, size*height), dpi=600)
    axes = axes.flatten()
    for i, case in enumerate(comparisons):
        tke_true_plot = case['results']['tke_true']
        true_idx = case['results'].get('true_idx', 0)
        tke_pred_plot = case['results']['tke_pred']
        pred_idx = case['results'].get('pred_idx', 0)
        axes[i].plot(np.array(true_idx) / 100, tke_true_plot, 'k-', label='Ground Truth')
        axes[i].plot(np.array(pred_idx) / 100, tke_pred_plot, linestyle=case['line_style'], label=case['label'], color=case.get('color', None))
        axes[i].set_title(case['label'])
        axes[i].set_xlabel('Nondimensional time')
        axes[i].set_ylabel(r'$\mathrm{TKE} = \frac{1}{2} \sum \mathbf{u}^2$')

        # axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axes[i].grid(visible=True, linestyle='--', linewidth=0.5)
        
        f, Pxx_pred = welch(case['results']['tke_pred'], fs=100, nperseg=256)
        axes[n_models].loglog(f, Pxx_pred, linestyle=case['line_style'], label=case['label'], color=case.get('color', None))

    axes[n_models].loglog(f, Pxx_true, 'k-', label='Ground Truth')
    axes[n_models].set_xlabel('Nondimensional Frequency')
    axes[n_models].set_ylabel('PSD($\mathrm{TKE}$)')
    axes[n_models].set_title('Power Spectral Density of TKE')

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
    plt.savefig(os.path.join(comp_config['save_dir'], 'TKE_comparison_subplots.png'), dpi=600)
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


    size = 0.75
    fig, axs = plt.subplots(2, 2, figsize=(size*width, size*height))
    # Coefficients for Point 1
    coeff_true_p1 = truth['p1'][time_lag:, :]
    kdeplot(coeff_true_p1[:, 0], ax=axs[0, 0], label='Ground Truth', color='k', common_norm=True)
    axs[0, 0].set_title('PDF($u_{p1}$)')

    kdeplot(coeff_true_p1[:, 1], ax=axs[0, 1], label='True $v_{p1}$', color='k', common_norm=True)
    axs[0, 1].set_title('PDF($v_{p1}$)')


    # Coefficients for Point 2
    coeff_true_p2 = truth['p2'][time_lag:, :]
    kdeplot(coeff_true_p2[:, 0], ax=axs[1, 0], label='True $u_{p2}$', color='k', common_norm=True)
    axs[1, 0].set_title('PDF($u_{p2}$)')

    kdeplot(coeff_true_p2[:, 1], ax=axs[1, 1], label='True $v_{p2}$', color='k', common_norm=True)
    axs[1, 1].set_title('PDF($v_{p2}$)')

    # Now plot predictions from each model
    for i, case in enumerate(comparisons):
        pred = case['results']['point_dict']['pred']
        time_lag = case['results'].get('time_lag', 0)

        coeff_pred_p1 = pred['p1'][time_lag:, :]
        kdeplot(coeff_pred_p1[:, 0], ax=axs[0, 0], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True)
        kdeplot(coeff_pred_p1[:, 1], ax=axs[0, 1], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True)

        coeff_pred_p2 = pred['p2'][time_lag:, :]
        kdeplot(coeff_pred_p2[:, 0], ax=axs[1, 0], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True)
        kdeplot(coeff_pred_p2[:, 1], ax=axs[1, 1], label=f"{case['label']}", linestyle=case['line_style'], color=case.get('color', None), common_norm=True)


    for ax in axs.flat:
        # ax.grid(visible=True, linestyle='--', linewidth=0.5)  
        ax.set_ylabel('')  
        ax.set_yticks([])
    # Collect all handles and labels from all axes
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
    plt.savefig(os.path.join(comp_config['save_dir'], 'PDF_comparison.png'), dpi=600)
    plt.close()
