### load in modules
import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from methods.models import Model
import numpy as np
import pandas as pd
import argparse
import ast
import myokit
from matplotlib import pyplot as plt

def parse_list_of_lists(s):
    try:
        # Convert the string representation of the list to an actual list
        return ast.literal_eval(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid list of lists: {s}") from e

### define parser
parser = argparse.ArgumentParser(description='Plot fits and log-likelihoods')
parser.add_argument('-p', type=str, required=True, help='.mmt protocol file')
parser.add_argument('-o', type=str, required=True, help='output folder for synthetic data')
parser.add_argument('-w', type=parse_list_of_lists, required=True, help='pulse windows')
parser.add_argument('-m', type=str, default="['7','10','11','12']", help='Model numbers')
parser.add_argument('-c', type=str, default="['bepridil','verapamil','terfenadine']", help='drug compounds')
args = parser.parse_args()

### define function that generates model output
def model_outputs(model_pars, prot, times, concs, wins):
    model_out = {}
    ### 5 x 2 pulses
    swps = 5
    win_1 = (times >= wins[0][0]) & (times < wins[0][1])
    win_2 = (times >= wins[1][0]) & (times < wins[1][1])
    for m in model_pars:
        binding_params = model_pars[m]
        model_out[m] = {}
        for conc in concs:
            model = Model(f'sis-m{m}',
                            prot,
                            parameters=['binding'],
                            analytical=True)
            # fix kt if necessary
            if m in ['12', '13']:
                model.fix_kt()
            try:
                # loop to simulate and append model output for # of sweeps 
                model_milnes = []
                model.set_dose(conc)
                after = model.simulate(binding_params, times)
                model_milnes = np.append(model_milnes, after[win_1])
                model_milnes = np.append(model_milnes, after[win_2])
                for i in range(swps-1):
                    after = model.simulate(binding_params, times, reset=False)
                    model_milnes = np.append(model_milnes, after[win_1])
                    model_milnes = np.append(model_milnes, after[win_2])
                model_out[m][conc] = model_milnes
            except:
                model_out[m][conc] = np.ones(times.shape) * float('inf')
    return model_out

### define drugbind class
def drugbind(prot_file, times, wins):
    '''
    For some input myokit protocol, this function generates synthetic current data under that protocol, 
    and then calculates pairwise ssq difference between a set of models. It then takes the median of these 
    and outputs this as an objective as well as the model outputs.
    '''
    # generate model output under the protocol
    model_out = []
    for mp, co in zip(model_pars_all, concs):
        model_out_t = model_outputs(mp, prot_file, times, co, wins)
        model_out = np.append(model_out, model_out_t)
    ### loop through models, drugs, and concentrations to get traces
    obj = 0
    for i,d in enumerate(model_out):
        all_traces = []
        for m in d:
            model_trace = []
            for c in concs[i]:
                model_trace = np.append(model_trace, d[m][c])
            all_traces.append(model_trace)
        ssq = [sum((m-n)**2) for i,m in enumerate(all_traces) for j,n in enumerate(all_traces) if i < j]
        obj += -np.median(ssq)
    return obj, model_out

### define drugs, and concentrations
d_list = ast.literal_eval(args.c)
concs = []
for drug in d_list:
    if drug == 'verapamil':
        concs.append([100,300,1000])
    elif drug == 'bepridil' or drug == 'terfenadine':
        concs.append([30,100,300])

### read model nums
model_nums = ast.literal_eval(args.m)

# get fitted drug-binding parameters
model_pars_all = []
for drug in d_list:
    model_pars = {}
    for j, m in enumerate(model_nums):
        df = pd.read_csv(f'{args.o}/{drug}/fits/{model_nums[j]}_fit_20000_points.csv')
        parstring = df.loc[df['score'].idxmax()]['pars']
        cleaned_string = parstring.replace("[", "").replace("]", "").replace("\n", "").strip()
        parlist = [float(i) for i in cleaned_string.split(" ") if i]
        model_pars[m] = parlist
    model_pars_all = np.append(model_pars_all, model_pars)

protocol_file = args.p
protocol = myokit.load_protocol(protocol_file)
max_time = protocol.characteristic_time()
times = np.arange(0, max_time, 10)
d_obj, model_outs = drugbind(protocol_file, times, args.w)

fig, axes = plt.subplots(len(concs[0]), len(d_list), figsize = (5*len(concs[0]), 3*len(d_list)))
for i,d in enumerate(model_outs):
    for m in d:
        for j, c in enumerate(concs[i]):
            model_trace = d[m][c]
            axes[i][j].plot(model_trace,label=f'Model {m}')
            axes[i][j].set_title(f'{c}nM {d_list[i]}')
            axes[i][j].set_ylim(bottom=-0.5,top=2.0)
plt.suptitle(f'Model currents (objective = {-np.round(d_obj,2)})')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{args.o}/{protocol_file.split("/")[-1].split(".")[0]}_model_split.png')

ts = protocol.log_for_times(np.arange(0, max_time, 10))
vs = protocol.value_at_times(np.arange(0, max_time, 10))
fig, ax = plt.subplots(figsize = (4,2))
ax.plot(ts['time'], vs)
ax.set_ylim(bottom = -130, top = 60)
plt.savefig(f'{args.o}/{protocol_file.split("/")[-1].split(".")[0]}.png')
