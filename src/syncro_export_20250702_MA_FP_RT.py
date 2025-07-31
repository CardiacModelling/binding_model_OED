### Import modules
import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from syncropatch_export.syncropatch_export.trace import Trace as tr
from pcpostprocess.pcpostprocess import leak_correct as lc
import pickle
import csv
import argparse
import skfda
from scipy.linalg import solve
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(description='Fitting synthetic data')
parser.add_argument('-o', type=str, required=True, help='output folder for synthetic data')
parser.add_argument('-d', type=str, required=True, help='drug compound string')
parser.add_argument('-n', action='store_true', help='load in new syncropatch data and export to pickle')
args = parser.parse_args()

concs_all = {'diltiazem': [3000,10000,30000],
             'chlorpromazine':[150,500,1500],
             'quinidine': [150,500,1500],
             'DMSO': [0]}

cols_all = {'diltiazem': [['02','03','04'],['05','06'],['07','08']],
            'chlorpromazine': [['10','11','12'],['13','14'],['15','16']],
            'quinidine': [['18','19','20'],['21','22'],['23','24']],
            'DMSO': [['01','09','17']]}

def fit_splines(t, x, n_order = 4, lambda_ = 5, knots = 6):

    knots = np.arange(min(t), max(t)+1, (max(t)-min(t))/(knots-1))

    bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=n_order)
    phi = np.array(bsplines(t)).T[0]

    # calculate penalty matrix
    operator = skfda.misc.operators.LinearDifferentialOperator(2)
    regularization = skfda.misc.regularization.L2Regularization(operator)
    r = regularization.penalty_matrix(bsplines)

    m = solve(phi.T @ phi + lambda_ * r, phi.T)
    c_hat = m @ x
    x_hat = phi @ c_hat

    return x_hat

def fit_lin_reg(t, x):
    model = LinearRegression()

    # Fit the model to the data
    model.fit(t, np.array(x))

    # Predict using the model
    x_pred = model.predict(t)

    return model.coef_[0], x_pred

def main(output_folder, drug, concs, cols):
    pulse = [2700, 22700]
    sweeps_oi = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    for c_j, conc in enumerate(concs):
            ### leak correct
            col = cols[c_j]
            curr_conc = {}
            gradcall = {}
            for well in currents.keys():
                if well[1:] in col and well in pass_list:
                    curr_conc[well] = {}
                    final_c = []
                    (b1i, b2i), ilki_fb = lc.fit_linear_leak(currents_fb[well][0], voltages, ts, 1700, 2500)
                    curr_conc[well][0] = currents_fb[well][0]-ilki_fb
                    (b1i, b2i), ilki = lc.fit_linear_leak(staircase_currents[well][0], staircase_voltages, staircase_ts, 600, 1400)
                    ilki = b2i*voltages + b1i
                    for i in sweeps_oi:
                        curr_conc[well][i] = currents[well][i]-ilki
                    for i in [3, 4, 5, 6, 7]:
                        cont_curr = currents[well][i]-ilki-curr_conc[well][0]
                        x_hat = fit_splines(ts[2700:22700], cont_curr[2700:22700], n_order = 4, lambda_ = 10**11, knots = 6)
                        final_c.append([x_hat[-1]])
                    gradc, x_fit = fit_lin_reg([[i] for i in [3, 4, 5, 6, 7]], final_c)
                    flat_section = curr_conc[well][7][:1600] - curr_conc[well][0][:1600]
                    noise = np.var(flat_section)
                    gradcall[well] = gradc[0]/noise
            ### additional QC
            wells_filt = ['P01','K07','M08','E10','E11','F11','P11','O13','C14','O14',
                          'E15','K15','B16','H16','N16','I19','I22','J22','N23']
            j = 0
            for well in list(curr_conc.keys()):
                if well not in wells_filt:
                    if gradcall[well] > 0.15 or gradcall[well] < -0.15:
                        if well not in wells_filt:
                            wells_filt+=[well]
                    j+=1

            fig,ax=plt.subplots(figsize = (10, 4))
            lc_fb_all = []
            lc_fb_full_all = []
            controls_all = []
            j = 0
            for well in list(curr_conc.keys()):
                if well not in wells_filt:
                    t_last = 0
                    t_last_full = 0
                    j+=1
                    fb = curr_conc[well][0]
                    control = curr_conc[well][7][pulse[0]:pulse[1]] - fb[pulse[0]:pulse[1]]
                    controls = []
                    lc_fbs = []
                    lc_fb_full = []
                    timesall = []
                    timesall_full = []
                    for i in sweeps_oi[1:]:
                        times = [t + t_last - ts[pulse[0]] for t in ts[pulse[0]:pulse[1]]]
                        lc_fb_sweep = (curr_conc[well][i][pulse[0]:pulse[1]] - fb[pulse[0]:pulse[1]])/control
                        lc_fbs += list(lc_fb_sweep)
                        lc_fb_full += list((curr_conc[well][i] - fb)/(curr_conc[well][7]-fb))
                        controls += list(control)
                        timesall += times
                        timesall_full += [t + t_last_full for t in ts]
                        t_last_full = timesall_full[-1] + 0.5
                        t_last = times[-1] + 0.5
                    lc_fb_all.append(lc_fbs)
                    lc_fb_full_all.append(lc_fb_full)
                    controls_all.append(controls)
            ax.plot(timesall, np.average(lc_fb_all, axis=0), linewidth = 0.01)
            ax.set_ylabel('prop. open')
            ax.set_ylim(bottom = -2, top = 3)
            ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5, alpha = 0.5)
            ax.set_title(f'{conc}nM {drug} sweeps 1-10 ({np.shape(lc_fb_all)[0]} wells)')
            fig.suptitle('Leak-corrected data w/ full block subtraction', fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{output_folder}/{conc}nM_block.png')
            with open(f"{output_folder}/fb_synthetic_conc_{conc}.csv", 'w') as f:
                f.write('"time","current"')
                f.write("\n")
                writer = csv.writer(f)
                writer.writerows(zip(timesall, np.average(lc_fb_all, axis=0)))

            with open(f"{output_folder}/fb_synthetic_conc_{conc}_full.csv", 'w') as f:
                f.write('"time","current"')
                f.write("\n")
                writer = csv.writer(f)
                writer.writerows(zip(timesall_full, np.average(lc_fb_full_all, axis=0)))

            # save control data for splinefitting
            dictY = {'t': timesall, 'x':  np.average(controls_all, axis=0)}
            pd.DataFrame(dictY).to_csv(f"{output_folder}/synth_Y.csv", index=False)

            # save standard deviation (in case not fitting)
            #sd_all = []
            #for c in controls_all:
            #    sd_all.append(np.sqrt(np.var(c[19000:20000])))
            #print(np.mean(sd_all))

if __name__ == "__main__":
    drug = args.d
    new_data=args.n
    output_folder = args.o
    concs = concs_all[drug]
    cols = cols_all[drug]
    colrs = [f'C{i}' for i in range(len(concs))]
    input_folder="Frankie_FastSlowDrugsMilnes_July_2025/20250702_MA_hERG_RT18T49769_Frankie"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(f"{output_folder}/fitting_output"):
        os.makedirs(f"{output_folder}/fitting_output")

    # control and drug sweeps
    filepath = f"{input_folder}/milnes_protocol_14.58.32/"
    json_file = "milnes_protocol_14.58.32"
    staircase_filepath = f"{input_folder}/Staircaserampx1_2kHz_14.57.40/"
    staircase_json_file = "Staircaserampx1_2kHz_14.57.40"
    test_trace = tr(filepath, json_file)
    staircase_test_trace = tr(staircase_filepath, staircase_json_file)
    voltages = test_trace.get_voltage()
    ts = test_trace.get_times()
    staircase_voltages = staircase_test_trace.get_voltage()
    staircase_ts = staircase_test_trace.get_times()

    # full block
    filepath_fb = f"{input_folder}/milnes_protocol_15.08.58/"
    json_file_fb = "milnes_protocol_15.08.58"
    test_trace_fb = tr(filepath_fb, json_file_fb)

    if new_data:
        currents = test_trace.get_all_traces(leakcorrect=False)
        staircase_currents = staircase_test_trace.get_all_traces(leakcorrect=False)
        with open(f'{input_folder}/{json_file}.pickle', 'wb') as handle:
            pickle.dump(currents, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{input_folder}/{staircase_json_file}.pickle', 'wb') as handle:
            pickle.dump(staircase_currents, handle, protocol=pickle.HIGHEST_PROTOCOL)
        currents_fb = test_trace_fb.get_all_traces(leakcorrect=False)
        with open(f'{input_folder}/{json_file_fb}.pickle', 'wb') as handle:
            pickle.dump(currents_fb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(f'{input_folder}/{json_file}.pickle', 'rb') as handle:
                currents = pickle.load(handle)
            with open(f'{input_folder}/{staircase_json_file}.pickle', 'rb') as handle:
                staircase_currents = pickle.load(handle)
            with open(f'{input_folder}/{json_file_fb}.pickle', 'rb') as handle:
                currents_fb = pickle.load(handle)
        except:
            print("Pickle file not found, try using argument '-n' to load new data")

    # QC1 and QC2
    with open(f'{input_folder}/selected-20250702_MA_hERG_RT18T49769_Frankie.txt', 'r') as file:
        pass_list = [line.strip() for line in file]

    main(output_folder, drug, concs, cols)
