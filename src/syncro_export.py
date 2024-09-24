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
parser.add_argument('-n', action='store_true', help='laod in new syncropatch data and export to pickle')
args = parser.parse_args()

concs_all = {'bepridil': [30,100,300],
             'quinidine':[150,500,1500],
             'verapamil': [100,300,1000]}

cols_all = {'bepridil': [['04','05','06'],['07','08'],['09','10']],
            'quinidine': [['11','12','13'],['14','15'],['16','17']],
            'verapamil': [['18','19','20'],['21','22'],['23','24']]}

### Define signal-to-noise QC function
def get_snr(cc_well, swp):
    flat_section = cc_well[swp][2*14500:2*17500] - cc_well[20][2*14500:2*17500]
    noise = np.var(flat_section)
    p1 = cc_well[swp][2*1350:2*4680] - cc_well[20][2*1350:2*4680]
    p2 = cc_well[swp][2*8010:2*11350] - cc_well[20][2*8010:2*11350]
    p3 = cc_well[swp][2*17930:2*27930] - cc_well[20][2*17930:2*27930]
    signal = list(p1) + list(p2) + list(p3)
    return sum(sn < (0-noise/2) for sn in signal)

def main(output_folder, drug, concs, cols):
    ### Perform QC1
    QCdf = test_trace.get_onboard_QC_df()
    QCdffilt = QCdf[(QCdf['Rseal']>0.1*10**9) & (QCdf['Rseal']<1000*10**9) & (QCdf['Cm']>1*10**-12) & (QCdf['Cm']<100*10**-12) &
        (QCdf['Rseries']>1*10**6) & (QCdf['Rseries']<25*10**6)]

    sweeps_oi = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
    for c_j, conc in enumerate(concs):
            ### leak correct
            col = cols[c_j]
            curr_conc = {}
            b1all = {}
            b2all = {}
            for well in currents.keys():
                if well[1:] in col and list(QCdffilt[QCdffilt['sweep'].isin(sweeps_oi)]['well']).count(well) == 12:
                    curr_conc[well] = {}
                    b1 = []
                    b2 = []
                    for i in sweeps_oi:
                        (b1i, b2i), ilki = lc.fit_linear_leak(currents[well][i], voltages, ts, 1700, 2500)
                        curr_conc[well][i] = currents[well][i]-ilki
                        b1 += [b1i]
                        b2 += [b2i]
                    b1all[well] = b1
                    b2all[well] = b2
            ### additional QC
            wells_filt = ['F06']
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
                    for i in np.arange(0, 10):
                        if i == 4:
                            if get_snr(curr_conc[well], i)>100:
                                if well not in wells_filt:
                                    wells_filt+=[well]
                        if get_snr(curr_conc[well], i+6)>300:
                            if well not in wells_filt:
                                wells_filt+=[well]

            fig,ax=plt.subplots(figsize = (10, 4))
            pulse1 = [1350, 4680]
            pulse2 = [8010, 11350]
            pulse3 = [17930, 27930]
            lc_fb_all = []
            controls_all = []
            j = 0
            for well in list(curr_conc.keys()):
                if well not in wells_filt:
                    t_last = 0
                    j+=1
                    fb = curr_conc[well][20]
                    control1 = curr_conc[well][4][2*pulse1[0]:2*pulse1[1]] - fb[2*pulse1[0]:2*pulse1[1]]
                    control2 = curr_conc[well][4][2*pulse2[0]:2*pulse2[1]] - fb[2*pulse2[0]:2*pulse2[1]]
                    control3 = curr_conc[well][4][2*pulse3[0]:2*pulse3[1]] - fb[2*pulse3[0]:2*pulse3[1]]
                    controls = []
                    lc_fbs = []
                    timesall = []
                    for i in np.arange(0, 10):
                        times1 = [t + t_last - ts[2*pulse1[0]] for t in ts[2*pulse1[0]:2*pulse1[1]]]
                        times2 = [t + times1[-1] - ts[2*pulse2[0]] for t in ts[2*pulse2[0]:2*pulse2[1]]]
                        times3 = [t + times2[-1] - ts[2*pulse3[0]] for t in ts[2*pulse3[0]:2*pulse3[1]]]
                        lc_fb_sweep1 = (curr_conc[well][i+6][2*pulse1[0]:2*pulse1[1]] - fb[2*pulse1[0]:2*pulse1[1]])/control1
                        lc_fb_sweep2 = (curr_conc[well][i+6][2*pulse2[0]:2*pulse2[1]] - fb[2*pulse2[0]:2*pulse2[1]])/control2
                        lc_fb_sweep3 = (curr_conc[well][i+6][2*pulse3[0]:2*pulse3[1]] - fb[2*pulse3[0]:2*pulse3[1]])/control3
                        lc_fbs += list(lc_fb_sweep1)+list(lc_fb_sweep2)+list(lc_fb_sweep3)
                        controls += list(control1)+list(control2)+list(control3)
                        timesall += times1+times2+times3
                        t_last = times3[-1]
                    lc_fb_all.append(lc_fbs)
                    controls_all.append(controls)
            ax.plot(timesall, np.average(lc_fb_all, axis=0))
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
    input_folder="Frankie_Experiments_Nov_Dec_2023/08122023_MW2"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(f"{output_folder}/fitting_output"):
        os.makedirs(f"{output_folder}/fitting_output")

    # control and drug sweeps
    filepath = f"{input_folder}/3_drug_protocol_14.34.49/"
    json_file = "3_drug_protocol_14.34.49"
    test_trace = tr(filepath, json_file)
    voltages = test_trace.get_voltage()
    ts = test_trace.get_times()

    if new_data:
        currents = test_trace.get_all_traces(leakcorrect=False)
        with open(f'{input_folder}/3_drug_protocol_14.34.49.pickle', 'wb') as handle:
            pickle.dump(currents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(f'{input_folder}/3_drug_protocol_14.34.49.pickle', 'rb') as handle:
                currents = pickle.load(handle)
        except:
            print("Pickle file not found, try using argument '-n' to load new data")

    main(output_folder, drug, concs, cols)
