# import modules
import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from methods import sweeps, sd, steps
import methods.funcs as funcs
import methods.parameters as parameters
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import csv
import argparse
import ast
import myokit

def parse_list_of_lists(s):
    try:
        # Convert the string representation of the list to an actual list
        return ast.literal_eval(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid list of lists: {s}") from e

parser = argparse.ArgumentParser(description='Fitting synthetic data')
parser.add_argument('-m', type=str, required=True, help='Model number')
parser.add_argument('-p', type=str, required=True, help='Protocol file name')
parser.add_argument('-t', type=float, default=15e3, help='Max time')
parser.add_argument('-b', type=parse_list_of_lists, default="[[1e3, 11e3]]", help='Protocol window(s) of interest')
parser.add_argument('-e', type=str, default='joey_sis', help='hERG model parameters')
parser.add_argument('-o', type=str, required=True, help='output folder for synthetic data')
parser.add_argument('-c', type=str, required=True, help='drug compound string')
args = parser.parse_args()

def plot_fig(X, Y, Z, prot, bounds, output_folder):
    '''
    plots data
    '''
    fig = plt.figure(figsize=(7, 2.25))
    if args.t == 15e3:
        swps = sweeps
        gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.4, 0.6], height_ratios=[1, 1], hspace=0.65, wspace = 0.45)
        ax4 = plt.subplot(gs[0, 2])
        ax5 = plt.subplot(gs[1, 2])
    else:
        swps = 5
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1.4], height_ratios=[1, 1], hspace=0.5, wspace = 0.45)
    ax3 = plt.subplot(gs[:, 0])

    swp_len=10e3
    times_all = np.arange(0, len(Y)/2, steps)

    if args.t == 15e3:
        ax4.plot([t-1000 for t in times_all], Y, alpha = 0.8, color = '#1E152A')
        for conc, col in zip(concs, colrs):
            ax4.plot([t-1000 for t in times_all], X[conc], alpha = 0.8, color = col)
            ax5.plot([t-1000 for t in times_all], Z[conc], alpha = 0.8, color = col)

        axes1=[ax4,ax5]
        ax4.set_xlim(left = 0, right=2000)
        ax4.axhline(y=0, color = 'grey', linestyle = '--')
        ax4.set_ylabel('Current (pA)', fontsize = 10)
        ax4.set_title(f'1/5 of pulse 1', fontsize = 10)
        ax4.set_xticks(np.array([0,1000,2000]))
        ax4.tick_params(labelbottom=False)
        ax5.set_xlim(left = 0, right=2000)
        ax5.set_ylim(top = 1.5, bottom = -0.2)
        ax5.set_yticks(np.array([0,0.5,1,1.5]))
        ax5.set_xticks(np.array([0,1000,2000]))
        ax5.axhline(y=0, color = 'grey', linestyle = '--')
        ax5.set_xlabel('Time (ms)', fontsize = 10)
        ax5.set_ylabel('Proportion open', fontsize = 10)

    xticks = []
    for i in np.arange(0, swps):
        for b in bounds:
            xticks.append(b[0] + swp_len/2 + i*len(Y)/(2*swps))
    xlims=[(xval-swp_len/2, xval+swp_len/2) for xval in xticks]

    inner1 = gridspec.GridSpecFromSubplotSpec(1, 10,
                        subplot_spec=gs[0, 1], wspace=0.3, hspace=0.1)
    axes2=[]
    for i, lim in zip(np.arange(0, 10), xlims):
        ax = plt.Subplot(fig, inner1[i])
        ax.plot(times_all, Y, alpha = 0.8, color = '#1E152A', label = 'control')
        for conc, col in zip(concs, colrs):
            ax.plot(times_all, X[conc], alpha = 0.8, label = f'{conc}nM', color = col)
        ax.set_xlim(lim[0], lim[1])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        if i != 0:
            ax.spines.left.set_visible(False)
            ax.tick_params(labeltop=False, left=False, top=False, right=False, labelright=False, labelleft=False)
        ax.set_xticks([(lim[1] + lim[0])/2], [i+1])
        d = .5
        kwargs = dict(marker=[(-.5, -d), (.5, d)], markersize=6,
            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax.tick_params(labelbottom=False)
        if i != 9:
            ax.plot([1], [0], transform=ax.transAxes, **kwargs)
        if i != 0:
            ax.plot([0], [0], transform=ax.transAxes, **kwargs)
        if i == 4:
            ax.set_title(f'    10x pulses', fontsize = 10)
        if i == 0:
            ax.set_ylabel('Current (pA)',fontsize = 10)
        fig.add_subplot(ax)
        if i == 0:
            axes2.append(ax)
            if args.t == 15e3:
                ax.legend(ncol = 5, bbox_to_anchor=(22.5, 2), fontsize = 10)
        if i == 9 and args.t != 15e3:
            ax.legend(ncol = 5, bbox_to_anchor=(1, 2), fontsize = 10)

    inner2 = gridspec.GridSpecFromSubplotSpec(1, 10,
                        subplot_spec=gs[1, 1], wspace=0.3, hspace=0.1)
    for i, lim in zip(np.arange(0, 10), xlims):
        ax = plt.Subplot(fig, inner2[i])
        for conc, col in zip(concs, colrs):
            ax.plot(times_all, Z[conc], alpha = 0.8, color = col)
        ax.set_xlim(lim[0], lim[1])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        if i != 0:
            ax.spines.left.set_visible(False)
            ax.tick_params(labeltop=False, left=False, top=False, right=False, labelright=False, labelleft=False)
        ax.set_xticks([(lim[1] + lim[0])/2], [i+1])
        if i != 9:
            ax.plot([1], [0], transform=ax.transAxes, **kwargs)
        if i != 0:
            ax.plot([0], [0], transform=ax.transAxes, **kwargs)
        if i == 4:
            ax.set_xlabel('    Pulse',fontsize = 10)
        if i == 0:
            ax.set_ylabel('Proportion open',fontsize = 10)
            axes2.append(ax)
        ax.set_yticks(np.array([0,0.5,1,1.5]))
        ax.set_ylim(top = 1.5, bottom = -0.2)
        fig.add_subplot(ax)

    protocol = myokit.load_protocol(prot)
    times = np.arange(0, len(Y)/(2*swps), steps)
    ax3.plot([t-1000 for t in times], protocol.log_for_times(times)['pace'], color = 'k', linewidth=2)
    ax3.set_ylabel('Voltage (mV)', fontsize = 10)
    ax3.set_ylim(bottom = -100, top = 50)
    ax3.set_xlim(left = -1000, right = args.t-1000)
    ax3.set_xlabel('Time (ms)', fontsize = 10)
    ax3.set_title('Voltage Protocol', fontsize = 10)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.tick_params(axis='both', which='minor', labelsize=10)
    if args.t == 15e3:
        fig.align_ylabels(axs=axes1)
    fig.align_ylabels(axs=axes2)
    plt.savefig(f'{output_folder}/synth_data.png', dpi=1200, bbox_inches='tight')

def main(m_sel, prot, max_time, bounds, herg, output_folder):
    # generate data
    synth_X, synth_Y, synth_Z, _, synth_Y_win, synth_Z_win, ts, ts_win = funcs.generate_data(
                                                            herg, drug_vals, prot, sd, max_time, bounds, m_sel, concs)
    # save data
    for conc in concs:
        with open(f"{output_folder}/fb_synthetic_conc_{conc}.csv", 'w') as f:
            f.write('"time","current"')
            f.write("\n")
            writer = csv.writer(f)
            writer.writerows(zip(ts_win, synth_Z_win[conc]))
        with open(f"{output_folder}/fb_synthetic_conc_{conc}_full.csv", 'w') as f:
            f.write('"time","current"')
            f.write("\n")
            writer = csv.writer(f)
            writer.writerows(zip(ts, synth_Z[conc]))

    # save control data for splinefitting
    dictY = {'t': ts_win, 'x': synth_Y_win}
    pd.DataFrame(dictY).to_csv(f"{output_folder}/synth_Y.csv", index=False)
    plot_fig(synth_X, synth_Y, synth_Z, prot, bounds, output_folder)

if __name__ == "__main__":
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    if not os.path.exists(f"{args.o}/fitting_output"):
        os.makedirs(f"{args.o}/fitting_output")

    if args.m in ['12','13']:
        drug_vals = parameters.binding[f'm{args.m}'][args.c][1:]
    else:
        drug_vals = parameters.binding[f'm{args.m}'][args.c]

    concs = parameters.drug_concs[args.c]
    colrs = [f'C{i}' for i in range(len(concs))]

    main(args.m, args.p, args.t, args.b, args.e, args.o)
