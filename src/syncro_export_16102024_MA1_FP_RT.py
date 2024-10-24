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

parser = argparse.ArgumentParser(description='Fitting synthetic data')
parser.add_argument('-o', type=str, required=True, help='output folder for synthetic data')
parser.add_argument('-d', type=str, required=True, help='drug compound string')
parser.add_argument('-n', action='store_true', help='load in new syncropatch data and export to pickle')
args = parser.parse_args()

concs_all = {'terfenadine': [30,100,300],
             'bepridil':[30,100,300],
             'verapamil': [100,300,1000],
             'DMSO': [1]}

cols_all = {'terfenadine': [['02','03','04'],['05','06'],['07','08']],
            'bepridil': [['10','11','12'],['13','14'],['15','16']],
            'verapamil': [['18','19','20'],['21','22'],['23','24']],
            'DMSO': [['01','09','17']]}

### Define signal-to-noise QC function
def get_snr(cc_well, swp):
    flat_section = cc_well[swp][26000:] - cc_well[0][26000:]
    noise = np.var(flat_section)
    signal = list(cc_well[swp][2700:22700] - cc_well[0][2700:22700])
    return sum(sn < (0-noise/2) for sn in signal)

def main(output_folder, drug, concs, cols):
    ### Perform QC1
    QCdf = test_trace.get_onboard_QC_df()
    QCdf_fb = test_trace_fb.get_onboard_QC_df()
    QCdffilt = QCdf[(QCdf['Rseal']>0.1*10**9) & (QCdf['Rseal']<1000*10**9) & (QCdf['Cm']>1*10**-12) & (QCdf['Cm']<100*10**-12) &
        (QCdf['Rseries']>1*10**6) & (QCdf['Rseries']<25*10**6)]
    QCdffilt_fb = QCdf_fb[(QCdf_fb['Rseal']>0.1*10**9) & (QCdf_fb['Rseal']<1000*10**9) & (QCdf_fb['Cm']>1*10**-12) & (QCdf_fb['Cm']<100*10**-12) &
     (QCdf_fb['Rseries']>1*10**6) & (QCdf_fb['Rseries']<25*10**6)]

    sweeps_oi = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    for c_j, conc in enumerate(concs):
            ### leak correct
            col = cols[c_j]
            curr_conc = {}
            b1all = {}
            b2all = {}
            for well in currents.keys():
                if well[1:] in col and list(QCdffilt[QCdffilt['sweep'].isin(sweeps_oi)]['well']).count(well) == 21 and list(QCdffilt_fb[QCdffilt_fb['sweep'].isin([0])]['well']).count(well) == 1:
                    curr_conc[well] = {}
                    b1 = []
                    b2 = []
                    for i in sweeps_oi:
                        (b1i, b2i), ilki = lc.fit_linear_leak(currents[well][i], voltages, ts, 1700, 2500)
                        curr_conc[well][i] = currents[well][i]-ilki
                        b1 += [b1i]
                        b2 += [b2i]
                    (b1i, b2i), ilki = lc.fit_linear_leak(currents_fb[well][0], voltages, ts, 1700, 2500)
                    curr_conc[well][0] = currents_fb[well][0]-ilki
                    b1 += [b1i]
                    b2 += [b2i]
                    b1all[well] = b1
                    b2all[well] = b2
            ### additional QC
            wells_filt = ['G12', 'O21', 'M22']
            j = 0
            for well in list(curr_conc.keys()):
                if well not in wells_filt:
                    if min(b1all[well]) <= -4:
                        if well not in wells_filt:
                            wells_filt+=[well]
                    if max(b1all[well]) >= 90:
                        if well not in wells_filt:
                            wells_filt+=[well]
                    j+=1
                    for i in sweeps_oi:
                        if i == 8:
                            if get_snr(curr_conc[well], i)>40:
                                if well not in wells_filt:
                                    wells_filt+=[well]
                        else:
                            if get_snr(curr_conc[well], i)>300:
                                if well not in wells_filt:
                                    wells_filt+=[well]

            fig,ax=plt.subplots(figsize = (10, 4))
            pulse = [2700, 22700]
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
                    control = curr_conc[well][8][pulse[0]:pulse[1]] - fb[pulse[0]:pulse[1]]
                    controls = []
                    lc_fbs = []
                    lc_fb_full = []
                    timesall = []
                    timesall_full = []
                    for i in sweeps_oi[1:]:
                        times = [t + t_last - ts[pulse[0]] for t in ts[pulse[0]:pulse[1]]]
                        lc_fb_sweep = (curr_conc[well][i][pulse[0]:pulse[1]] - fb[pulse[0]:pulse[1]])/control
                        lc_fbs += list(lc_fb_sweep)
                        lc_fb_full += list((curr_conc[well][i] - fb)/(curr_conc[well][8]-fb))
                        controls += list(control)
                        timesall += times
                        timesall_full += [t + t_last_full for t in ts]
                        t_last_full = timesall_full[-1] + 0.5
                        t_last = times[-1] + 0.5
                    lc_fb_all.append(lc_fbs)
                    lc_fb_full_all.append(lc_fb_full)
                    controls_all.append(controls)
            ax.plot(timesall, np.average(lc_fb_all, axis=0))
            ax.set_ylabel('prop. open')
            ax.set_ylim(bottom = -2, top = 3)
            ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5, alpha = 0.5)
            ax.set_title(f'{conc}nM {drug} sweeps 1-20 ({np.shape(lc_fb_all)[0]} wells)')
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
    input_folder="Frankie_Experiments_Oct_Nov_2024/16102024_MA1_FP_RT"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(f"{output_folder}/fitting_output"):
        os.makedirs(f"{output_folder}/fitting_output")

    # control and drug sweeps
    filepath = f"{input_folder}/milnes_protocol_14.58.16/"
    json_file = "milnes_protocol_14.58.16"
    test_trace = tr(filepath, json_file)
    voltages = test_trace.get_voltage()
    ts = test_trace.get_times()

    # full block
    filepath_fb = f"{input_folder}/milnes_protocol_15.16.32/"
    json_file_fb = "milnes_protocol_15.16.32"
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
