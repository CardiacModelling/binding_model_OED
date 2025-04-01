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
             'DMSO': [1]}

cols_all = {'diltiazem': [['02','03','04'],['05','06'],['07','08']],
            'chlorpromazine': [['10','11','12'],['13','14'],['15','16']],
            'quinidine': [['18','19','20'],['21','22'],['23','24']],
            'DMSO': [['01','09','17']]}

### Define signal-to-noise QC function
def get_snr(cc_well, swp):
    flat_section = cc_well[swp][:1600] - cc_well[0][:1600]
    noise = np.var(flat_section)
    signal = list(cc_well[swp][2700:22700] - cc_well[0][2700:22700])
    return sum(sn < (0-noise/2) for sn in signal)

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
    ### Perform QC1
    QCdf = test_trace.get_onboard_QC_df()
    QCdf_fb = test_trace_fb.get_onboard_QC_df()
    QCdffilt = QCdf[(QCdf['Rseal']>0.1*10**9) & (QCdf['Rseal']<1000*10**9) & (QCdf['Cm']>1*10**-12) & (QCdf['Cm']<100*10**-12) &
        (QCdf['Rseries']>1*10**6) & (QCdf['Rseries']<25*10**6)]
    QCdffilt_fb = QCdf_fb[(QCdf_fb['Rseal']>0.1*10**9) & (QCdf_fb['Rseal']<1000*10**9) & (QCdf_fb['Cm']>1*10**-12) & (QCdf_fb['Cm']<100*10**-12) &
     (QCdf_fb['Rseries']>1*10**6) & (QCdf_fb['Rseries']<25*10**6)]

    pulse = [2700, 22700]
    sweeps_oi = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for c_j, conc in enumerate(concs):
            ### leak correct
            col = cols[c_j]
            curr_conc = {}
            b1all = {}
            gradcall = {}
            for well in currents.keys():
                if well[1:] in col and list(QCdffilt[QCdffilt['sweep'].isin(sweeps_oi)]['well']).count(well) == 11 and list(QCdffilt_fb[QCdffilt_fb['sweep'].isin([0])]['well']).count(well) == 1:
                    curr_conc[well] = {}
                    b1 = []
                    final_c = []
                    (b1i, b2i), ilki = lc.fit_linear_leak(currents_fb[well][0], voltages, ts, 1700, 2500)
                    curr_conc[well][0] = currents_fb[well][0]-ilki
                    for i in sweeps_oi:
                        if i == 6:
                            (b1i, b2i), ilki = lc.fit_linear_leak(currents[well][0], voltages, ts, 1700, 2500)
                        else:
                            (b1i, b2i), ilki = lc.fit_linear_leak(currents[well][7], voltages, ts, 1700, 2500)
                        curr_conc[well][i] = currents[well][i]-ilki
                        b1 += [b1i]
                    for i in [2, 3, 4, 5, 6]:
                        (b1i, b2i), ilki = lc.fit_linear_leak(currents[well][0], voltages, ts, 1700, 2500)
                        cont_curr = currents[well][i]-ilki-curr_conc[well][0]
                        x_hat = fit_splines(ts[2700:22700], cont_curr[2700:22700], n_order = 4, lambda_ = 10**11, knots = 6)
                        final_c.append([x_hat[-1]])
                    gradc, x_fit = fit_lin_reg([[i] for i in [2, 3, 4, 5, 6]], final_c)
                    (b1i, b2i), ilki = lc.fit_linear_leak(currents_fb[well][0], voltages, ts, 1700, 2500)
                    curr_conc[well][0] = currents_fb[well][0]-ilki
                    flat_section = curr_conc[well][6][:1600] - curr_conc[well][0][:1600]
                    noise = np.var(flat_section)
                    b1 += [b1i]
                    gradcall[well] = gradc[0]/noise
                    b1all[well] = [b/noise for b in b1]
            ### additional QC
            wells_filt = ['A07','A11','B07','B14','C22','D14','E05','E21','E24','G18',
                          'J11','J17','K02','K24','L18','L22','M19','N04','N11','N13',
                          'A17','A24','B03','B04','B13','B16','B24','C03','C06','C07',
                          'C11','D01','E09','E13','E18','E19','F09','F16','G02','G15',
                          'G16','G17','H02','H10','H17','I01','I09','I11','I13','J04',
                          'J09','J22','J24','K04','K05','K18','L03','L15','L24','M18',
                          'M12','M20','M22','M24','N01','N02','N05','O08','O11','P01',
                          'P11','P13','P15','P24','F01','A15']
            j = 0
            for well in list(curr_conc.keys()):
                if well not in wells_filt:
                    if gradcall[well] > 0.1 or gradcall[well] < -0.1:
                        if well not in wells_filt:
                            wells_filt+=[well]
                    if min(b1all[well]) <= -0.1:
                        if well not in wells_filt:
                            wells_filt+=[well]
                    j+=1
                    for i in sweeps_oi:
                        if i == 6:
                            if get_snr(curr_conc[well], i)>100:
                                if well not in wells_filt:
                                    wells_filt+=[well]
                        else:
                            if get_snr(curr_conc[well], i)>200:
                                if well not in wells_filt:
                                    wells_filt+=[well]

            for i in ['C05', 'P07', 'K10', 'I12', 'M16', 'P20', 'F22', 'G21', 'C23', 'F23']:
                if i in wells_filt:
                    wells_filt.remove(i)

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
                    control = curr_conc[well][6][pulse[0]:pulse[1]] - fb[pulse[0]:pulse[1]]
                    controls = []
                    lc_fbs = []
                    lc_fb_full = []
                    timesall = []
                    timesall_full = []
                    for i in sweeps_oi[1:]:
                        times = [t + t_last - ts[pulse[0]] for t in ts[pulse[0]:pulse[1]]]
                        lc_fb_sweep = (curr_conc[well][i][pulse[0]:pulse[1]] - fb[pulse[0]:pulse[1]])/control
                        lc_fbs += list(lc_fb_sweep)
                        lc_fb_full += list((curr_conc[well][i] - fb)/(curr_conc[well][6]-fb))
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

if __name__ == "__main__":
    drug = args.d
    new_data=args.n
    output_folder = args.o
    concs = concs_all[drug]
    cols = cols_all[drug]
    colrs = [f'C{i}' for i in range(len(concs))]
    input_folder="Frankie_Experiments_Oct_Nov_2024/30102024_MA_FP_RT"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(f"{output_folder}/fitting_output"):
        os.makedirs(f"{output_folder}/fitting_output")

    # control and drug sweeps
    filepath = f"{input_folder}/milnes_protocol_14.06.08/"
    json_file = "milnes_protocol_14.06.08"
    test_trace = tr(filepath, json_file)
    voltages = test_trace.get_voltage()
    ts = test_trace.get_times()

    # full block
    filepath_fb = f"{input_folder}/milnes_protocol_14.21.07/"
    json_file_fb = "milnes_protocol_14.21.07"
    test_trace_fb = tr(filepath_fb, json_file_fb)

    if new_data:
        currents = test_trace.get_all_traces(leakcorrect=False)
        with open(f'{input_folder}/{json_file}.pickle', 'wb') as handle:
            pickle.dump(currents, handle, protocol=pickle.HIGHEST_PROTOCOL)
        currents_fb = test_trace_fb.get_all_traces(leakcorrect=False)
        with open(f'{input_folder}/{json_file_fb}.pickle', 'wb') as handle:
            pickle.dump(currents_fb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(f'{input_folder}/{json_file}.pickle', 'rb') as handle:
                currents = pickle.load(handle)
            with open(f'{input_folder}/{json_file_fb}.pickle', 'rb') as handle:
                currents_fb = pickle.load(handle)
        except:
            print("Pickle file not found, try using argument '-n' to load new data")

    main(output_folder, drug, concs, cols)
