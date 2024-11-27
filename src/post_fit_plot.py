# import modules
import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from methods import steps, all_model_nums
import methods.funcs as funcs
import methods.parameters as parameters
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import ast

def parse_list_of_lists(s):
    try:
        # Convert the string representation of the list to an actual list
        return ast.literal_eval(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid list of lists: {s}") from e

parser = argparse.ArgumentParser(description='Plot fits and log-likelihoods')
parser.add_argument('-m', type=str, required=True, help='Model numbers')
parser.add_argument('-p', type=str, required=True, help='Protocol file name')
parser.add_argument('-t', type=float, default=15e3, help='Max time')
parser.add_argument('-b', type=parse_list_of_lists, default="[[1e3, 11e3]]", help='Protocol window(s) of interest')
parser.add_argument('-e', type=str, default='joey_sis', help='hERG model parameters')
parser.add_argument('-o', type=str, required=True, help='output folder for synthetic data')
parser.add_argument('-c', action='store_true', help='enable parameter and log-likelihood comparison plots')
parser.add_argument('-d', type=str, help='drug compound')
parser.add_argument('-s', type=int, default=10, help='no. of sweeps')
parser.add_argument('-y', action='store_true', help='enable plotting for case where both milnes and opt data have been fitted')
args = parser.parse_args()

def main(model_nums, prot, max_time, bounds, herg, output_folder, swps):
    # TODO hardcoded to determine sweep length
    if max_time == 15e3 or max_time == 25350:
        swp_len=10e3
    elif herg == '2024_Joey_sis_25C' and prot != "protocols/3_drug_protocol_23_10_24.mmt" and prot != "protocols/3_drug_protocol_14_11_24.mmt":
        swp_len=[3340, 3330, 10e3]
    else:
        swp_len=[]
        for b in bounds:
            swp_len.append(b[1]-b[0])
    # define simulation time
    times = np.arange(0, max_time, steps)
    wins = []
    for b in bounds:
        wins.append((times >= b[0]) & (times < b[-1]))

    length = 0
    for win in wins:
       length += len(times[win])

    # get fitted drug-binding parameters
    drug_fit_pars = {}
    drug_fit_score = []
    for j, m in enumerate(model_nums):
        if args.y and prot != "protocols/Milnes_16102024_MA1_FP_RT.mmt":
            df = pd.read_csv(f'{output_folder}/fits_milnes_and_opt/{model_nums[j]}_fit_{length}_points.csv')
        elif args.y:
            df = pd.read_csv(f'outputs_real_20241114_MA_FP_RT/{args.d}/fits_milnes_and_opt/{model_nums[j]}_fit_12000_points.csv')
        else:
            df = pd.read_csv(f'{output_folder}/fits/{model_nums[j]}_fit_{length}_points.csv')
        parstring = df.loc[df['score'].idxmax()]['pars']
        cleaned_string = parstring.replace("[", "").replace("]", "").replace("\n", "").strip()
        parlist = [float(i) for i in cleaned_string.split(" ") if i]
        drug_fit_pars[m] = parlist
        drug_fit_score.append(max(df['score']))

    # get model output for fitted parameters
    synth_Zfit_all = {}
    synth_Xfit_all = {}
    for m in all_model_nums:
        if m in model_nums:
            drug_vals = drug_fit_pars[m]
            synth_Xfit, synth_Yfit, synth_Zfit, _, _, _, ts, _ = funcs.generate_data(herg, drug_vals, prot, 0, max_time, bounds, m, concs, swps, notrecord=True)
            synth_Zfit_all[m] = synth_Zfit
            synth_Xfit_all[m] = synth_Xfit

    # get xticks and xlims
    xticks = []
    for i in np.arange(0, swps):
        s_j = 0
        for b in bounds:
            if herg != '2024_Joey_sis_25C' and max_time != 15e3:
                xticks.append(b[0] + swp_len[s_j]/2 + i*len(synth_Yfit)/(2*swps))
                s_j+=1
            elif herg != '2024_Joey_sis_25C' or max_time == 25350:
                xticks.append(b[0] + swp_len/2 + i*len(synth_Yfit)/(2*swps))
            else:
                xticks.append(b[0] + swp_len[s_j]/2 + i*len(synth_Yfit)/(2*swps))
                s_j+=1
    if max_time == 15e3 or max_time == 25350:
        xlims=[(xval-swp_len/2, xval+swp_len/2) for xval in xticks]
    elif herg == '2024_Joey_sis_25C':
        xlims=[]
        s_j = 0
        for xval in xticks:
            xlims.append((xval-swp_len[s_j]/2, xval+swp_len[s_j]/2))
            if s_j < 1:
                s_j+=1
            else:
                s_j=0
    else:
        xlims=[]
        for xval in xticks:
            s_j = 0
            xlims.append((xval-swp_len[s_j]/2, xval+swp_len[s_j]/2))
            if s_j < 1:
                s_j+=1
            else:
                s_j=0

    # Create a 5x3 grid of subplots of model fits
    fig = plt.figure(figsize=(14, 7))
    outer = gridspec.GridSpec(5, 3, wspace=0.15, hspace=0.45)
    for j, m in enumerate(all_model_nums):
        if m in model_nums:
            inner = gridspec.GridSpecFromSubplotSpec(1, 6,
                                subplot_spec=outer[j], wspace=0.3)
            for i, lim in zip(np.arange(0, 6), xlims):
                ax = plt.Subplot(fig, inner[i])
                for conc, col in zip(concs, colrs):
                    dfconc = pd.read_csv(f"{output_folder}/fb_synthetic_conc_{conc}_full.csv")
                    ax.plot(dfconc['time'], dfconc['current'], alpha = 0.2, color = col)
                    if i == 0 and j == 2:
                        ax.plot(dfconc['time'], synth_Zfit_all[m][conc], alpha = 0.8, color = col, label = f'{conc}nM')
                    else:
                        ax.plot(dfconc['time'], synth_Zfit_all[m][conc], alpha = 0.8, color = col)
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
                if i != 9:
                    ax.plot([1], [0], transform=ax.transAxes, **kwargs)
                if i != 0:
                    ax.plot([0], [0], transform=ax.transAxes, **kwargs)
                ax.set_ylim(top = 1.5, bottom = -0.2)
                ax.set_yticks(np.array([0,0.5,1,1.5]))
                if i == 4:
                    ax.set_title(f'    Model {m}', fontsize = 10)
                    if j in [12, 13, 14]:
                        ax.set_xlabel('    Pulse', fontsize = 10)
                if i == 0 and (j in [0, 3, 6, 9, 12]):
                    ax.set_ylabel('Proportion\n open', fontsize = 10)
                fig.add_subplot(ax)
                if i == 0 and j == 2:
                    ax.legend(bbox_to_anchor=(7.5, 1.35), fontsize = 10, ncol = 4)
                if j not in [12, 13, 14]:
                    ax.tick_params(labelbottom=False)
                if j in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]:
                    ax.tick_params(labelleft=False)
    if args.y:
        plt.savefig(f"{output_folder}/model_fits_milnes_and_opt.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(f"{output_folder}/model_fits.png", dpi=600, bbox_inches='tight')

    if args.c:
        if max_time == 15e3:
            # get fitted drug-binding parameters
            drug_fit_pars_non_opt = {}
            drug_fit_score_non_opt = []
            non_opt_model_nums = []
            for j, m in enumerate(model_nums):
                try:
                    df = pd.read_csv(f'{output_folder.split("/")[0]}/{output_folder.split("/")[1]}/{output_folder.split("/")[2]}/fits/{model_nums[j]}_fit_{int(length/2)}_points.csv')
                    parstring = df.loc[df['score'].idxmax()]['pars']
                    cleaned_string = parstring.replace("[", "").replace("]", "").replace("\n", "").strip()
                    parlist = [float(i) for i in cleaned_string.split(" ") if i]
                    drug_fit_pars_non_opt[m] = parlist
                    drug_fit_score_non_opt.append(max(df['score']))
                    non_opt_model_nums.append(m)
                except:
                    print(f"Model {m} was not fitted in the non-optimal case")
            # plot log-likelihoods
            fig = plt.figure(figsize=(7, 2))
            gs = gridspec.GridSpec(1, 2, wspace = 0.045)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax1.scatter(non_opt_model_nums,drug_fit_score_non_opt,marker = 'x',color = '#1E152A')
            ax2.scatter(all_model_nums,drug_fit_score,marker = 'x',color = '#1E152A')
            ax1.set_ylabel('Maximised log-likelihood',fontsize=10)
            ax1.set_xlabel('Fitted model',fontsize=10)
            ax2.set_xlabel('Fitted model',fontsize=10)
            ax2.tick_params(labelleft=False)
            ax1.axhline(np.max(drug_fit_score_non_opt)-10000, color = 'g', linestyle = '--', alpha = 0.5, label = '$10^4$ below max.')
            ax1.axhline(np.max(drug_fit_score_non_opt)-100000, color = 'r', linestyle = '--', alpha = 0.5, label = '$10^5$ below max.')
            ax2.axhline(np.max(drug_fit_score)-10000, color = 'g', linestyle = '--', alpha = 0.5, label = '$10^4$ below max.')
            ax2.axhline(np.max(drug_fit_score)-100000, color = 'r', linestyle = '--', alpha = 0.5, label = '$10^5$ below max.')
            ax1.axvspan(7.5, 8.5, alpha=0.2, color='grey', label = 'data-generating model')
            ax2.axvspan(7.5, 8.5, alpha=0.2, color='grey', label = 'data-generating model')
            minx = np.min([np.min(drug_fit_score), np.min(drug_fit_score_non_opt)])
            maxx = np.max([np.max(drug_fit_score), np.max(drug_fit_score_non_opt)])
            ax1.set_ylim(bottom = minx-100000, top = maxx+100000)
            ax2.set_ylim(bottom = minx-100000, top = maxx+100000)
            ax1.tick_params(axis='both', which='major', labelsize=8.5)
            ax2.tick_params(axis='both', which='major', labelsize=8.5)
            ax1.set_title('Milnes', fontsize=10)
            ax2.set_title('Optimised Protocol', fontsize=10)
            plt.savefig(f"{output_folder}/loglikelihoods.png", dpi=600, bbox_inches='tight')
        else:
            # plot log-likelihoods
            fig, ax = plt.subplots(figsize=(7, 2))
            ax.scatter(all_model_nums,drug_fit_score,marker = 'x',color = '#1E152A')
            ax.set_ylabel('Maximised log-likelihood',fontsize=10)
            ax.set_xlabel('Fitted model',fontsize=10)
            ax.axhline(np.max(drug_fit_score)-10000, color = 'g', linestyle = '--', alpha = 0.5, label = '$10^4$ below max.')
            ax.axhline(np.max(drug_fit_score)-100000, color = 'r', linestyle = '--', alpha = 0.5, label = '$10^5$ below max.')
            minx = np.min(drug_fit_score)
            maxx = np.max(drug_fit_score)
            ax.set_ylim(bottom = minx-100000, top = maxx+100000)
            ax.tick_params(axis='both', which='major', labelsize=8.5)
            ax.set_title('Optimised Protocol', fontsize=10)
            if args.y:
                plt.savefig(f"{output_folder}/loglikelihoods_milnes_and_opt.png", dpi=600, bbox_inches='tight')
            else:
                plt.savefig(f"{output_folder}/loglikelihoods.png", dpi=600, bbox_inches='tight')

        if max_time == 15e3:
            if model_nums == non_opt_model_nums:
                # plot parameter comparison
                fig,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15))= plt.subplots(3,5,figsize=(7,4))
                for m, ax in zip(all_model_nums,[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]):
                    if m in all_model_nums:
                        if m not in ['1', '5', '9']:
                            ax.tick_params(labelleft=False)
                        if m not in ['9', '10', '11', '12', '13']:
                            ax.tick_params(labelbottom=False)
                        ax.plot(np.arange(0, 1e10, 1e9), np.arange(0, 1e10, 1e9), linestyle = '--', color = 'k', alpha = 0.25)
                        ax.scatter(drug_fit_pars[m][:-1], drug_fit_pars_non_opt[m][:-1],  marker = 'x', s = 40)
                        ax.set_xlim(left = 1e-10, right = 1e10)
                        ax.set_ylim(bottom = 1e-10, top = 1e10)
                        ax.set_yscale('log')
                        ax.set_xscale('log')
                        ax.text(0.05, 0.95, m, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top')
                        if ax in [ax1, ax6, ax11]:
                            ax.set_ylabel('Milnes', fontsize = 10)
                        if ax in [ax11, ax12, ax13, ax14, ax15]:
                            ax.set_xlabel('Optimised', fontsize = 10)
                        ax.tick_params(axis='both', which='major', labelsize=10)
                        ax.tick_params(axis='both', which='minor', labelsize=10)
                        ax.set_xticks([1e-9, 1, 1e9])
                        ax.set_xticklabels(['1e-9', '1', '1e9'])
                        ax.set_yticks([1e-9, 1, 1e9])
                        ax.set_yticklabels(['1e-9', '1', '1e9'])
                plt.tight_layout()
                plt.savefig(f"{output_folder}/parameter_comparison.png", dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    if args.e != '2024_Joey_sis_25C':
        concs = parameters.drug_concs[args.d]
    elif args.d == 'bepridil':
        concs = [30, 100, 300]
    elif args.d == 'quinidine':
        concs = [150, 500, 1500]
    elif args.d == 'terfenadine':
        concs = [30, 100, 300]
    elif args.d == 'verapamil':
        concs = [100, 300, 1000]
    elif args.d == 'diltiazem':
        concs = [3000, 10000, 30000]
    elif args.d == 'chlorpromazine':
        concs = [150, 500, 1500]
    elif args.d == 'DMSO':
        concs = [1]
    colrs = [f'C{i}' for i in range(len(concs))]
    m_list = ast.literal_eval(args.m)
    main(m_list, args.p, args.t, args.b, args.e, args.o, args.s)
