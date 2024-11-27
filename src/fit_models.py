import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pints
from methods import steps
import methods.boundaries as boundaries
import methods.classes as classes
import methods.parameters as parameters
import numpy as np
import argparse
import csv
import pandas as pd
import ast
import math

def parse_list_of_lists(s):
    try:
        # Convert the string representation of the list to an actual list
        return ast.literal_eval(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid list of lists: {s}") from e

### Define Parser
# Check input arguments
parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-r', '--repeats', type = int,
        help='Number of optimisation runs from different initial guesses')
parser.add_argument('-m', '--model', type = str, help='Binding model for optimisation')
parser.add_argument('-p', '--params', type= str, help='hERG model parameters')
parser.add_argument('-v', '--protocol', type = str, help='Voltage protocol')
parser.add_argument('-o', '--output', type = str, help='Output folder')
parser.add_argument('-t', type=float, default=15e3, help='Max time')
parser.add_argument('-b', type=parse_list_of_lists, default="[[1e3, 11e3]]", help='Protocol window(s) of interest')
parser.add_argument('-c', type = str, help='Drug compound string')
parser.add_argument('-d', action='store_true', help='Enable dual fitting of Milnes and optimal protocol data')
args = parser.parse_args()

def get_pars(model_num):
    ### Set individual errors and weights
    likelihoods = []
    for conc in concs:
        # Create forward models
        if herg_model != 'kemp' and herg_model != '2024_Joey_sis_25C':
            model = classes.ConcatMilnesModel(f'm{model_num}', protocol, times,
                                          win, conc, param_dict)
            if args.d:
                if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt":
                    model_m = classes.ConcatMilnesModel(f'm{model_num}', 'protocols/Milnes_Phil_Trans.mmt', times_m,
                                          win_m, conc, param_dict)
                else:
                    model_m = classes.ConcatMilnesModel(f'm{model_num}', 'protocols/Milnes_16102024_MA1_FP_RT.mmt', times_m,
                                          win_m, conc, param_dict)
        elif herg_model == 'kemp':
            model = classes.ConcatMilnesModel(f'kemp-m{model_num}', protocol, times,
                                          win, conc, param_dict)
            if args.d:
                model_m = classes.ConcatMilnesModel(f'kemp-m{model_num}', 'protocols/Milnes_Phil_Trans.mmt', times_m,
                                          win_m, conc, param_dict)
        elif herg_model == '2024_Joey_sis_25C':
            model = classes.ConcatMilnesModel(f'sis-m{model_num}', protocol, times,
                                          win, conc, param_dict)
            if args.d:
                if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt":
                    model_m = classes.ConcatMilnesModel(f'sis-m{model_num}', 'protocols/Milnes_Phil_Trans.mmt', times_m,
                                          win_m, conc, param_dict)
                else:
                    model_m = classes.ConcatMilnesModel(f'sis-m{model_num}', 'protocols/Milnes_16102024_MA1_FP_RT.mmt', times_m,
                                          win_m, conc, param_dict)
        # Load data
        u = np.loadtxt(
            f'{outdir}/fb_synthetic_conc_{conc}.csv',
            delimiter=',',
            skiprows=1
        )
        concat_time = u[:, 0]
        concat_milnes = u[:, 1]
        if args.d:
            if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt":
                # Load data
                u_m = np.loadtxt(
                    f'{outdir.rsplit("/",1)[0]}/fb_synthetic_conc_{conc}.csv',
                    delimiter=',',
                    skiprows=1
                )
            elif protocol == "protocols/3_drug_protocol_23_10_24.mmt":
                # Load data
                u_m = np.loadtxt(
                    f'outputs_real_16102024_MA1_FP_RT/{args.c}/fb_synthetic_conc_{conc}.csv',
                    delimiter=',',
                    skiprows=1
                )
            elif protocol == "protocols/3_drug_protocol_14_11_24.mmt":
                # Load data
                u_m = np.loadtxt(
                    f'outputs_real_30102024_MA_FP_RT/{args.c}/fb_synthetic_conc_{conc}.csv',
                    delimiter=',',
                    skiprows=1
                )
            concat_time_m = u_m[:, 0]
            concat_milnes_m = u_m[:, 1]
        # Create single output problem
        problem = pints.SingleOutputProblem(model, concat_time, concat_milnes)
        likelihoods.append(classes.NormalRatioLogLikelihood(problem, mu_y))
        if args.d:
            problem_m = pints.SingleOutputProblem(model_m, concat_time_m, concat_milnes_m)
            likelihoods.append(classes.NormalRatioLogLikelihood(problem_m, mu_y_m))

    if len(likelihoods) > 1:
        f = pints.SumOfIndependentLogPDFs(likelihoods)
    else:
        f = likelihoods[0]
    bounds = boundaries.Boundaries(model_num, fix_hill=False, likelihood=True)

    # Fix random seed for reproducibility
    np.random.seed(101)
    # Transformation
    if model_num in ['12', '13']:
        transform = pints.ComposedTransformation(
            pints.LogTransformation(f.n_parameters() - 1),
            pints.IdentityTransformation(1),
        )
    else:
        transform = pints.LogTransformation(f.n_parameters())

    params, scores = [], []
    for i in range(repeats):
        q0 = bounds.sample()
        print(f(q0))
        while math.isnan(f(q0)) or math.isinf(f(q0)):
            q0 = bounds.sample()
            print(f'resample: {f(q0)}')
        # Create optimiser
        opt = pints.OptimisationController(
                f, q0, boundaries=bounds, transformation=transform, method=pints.CMAES)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(iterations=100, threshold=100)
        # Run optimisation
        try:
            with np.errstate(all='ignore'): # Tell numpy not to issue warnings
                p, s = opt.run()
                params.append(p)
                scores.append(s)
        except ValueError:
            import traceback
            traceback.print_exc()

    return params, scores

if __name__ == "__main__":
    ### Set drug, concentrations, and no. repeats
    repeats = args.repeats
    model_num = args.model
    p_path = args.params
    prot = args.protocol
    output = args.output
    max_time = args.t
    bounds = args.b
    herg_model = p_path.split(".")[0]
    if herg_model != '2024_Joey_sis_25C':
        concs = parameters.drug_concs[args.c]
    elif args.c == 'bepridil':
        concs = [30, 100, 300]
    elif args.c == 'quinidine':
        concs = [150, 500, 1500]
    elif args.c == 'terfenadine':
        concs = [30, 100, 300]
    elif args.c == 'verapamil':
        concs = [100, 300, 1000]
    elif args.c == 'diltiazem':
        concs = [3000, 10000, 30000]
    elif args.c == 'chlorpromazine':
        concs = [150, 500, 1500]
    elif args.c == 'DMSO':
        concs = [1]
    if herg_model != 'kemp' and herg_model != '2024_Joey_sis_25C':
        with open(f'methods/models/params/{p_path}', newline='') as csvfile:
            p_reader = csv.reader(csvfile)
            param_dict = {rows[0]:float(rows[1]) for rows in p_reader}
    else:
        param_dict = {}

    ### Set protocol, outdir and sweep window
    protocol = prot
    outdir = output
    times = np.arange(0, max_time, steps)
    conditions = []
    for b in bounds:
        conditions.append(((times >= b[0]) & (times < b[-1])))
    win = np.zeros_like(conditions[0], dtype=bool)
    for condition in conditions:
        win |= condition
    if args.d:
        if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt":
            times_m = np.arange(0, 15e3, steps)
            conditions_m = []
            for b in [[1e3, 11e3]]:
                conditions_m.append(((times_m >= b[0]) & (times_m < b[-1])))
        else:
            times_m = np.arange(0, 15.35e3, steps)
            conditions_m = []
            for b in [[1.35e3, 11.35e3]]:
                conditions_m.append(((times_m >= b[0]) & (times_m < b[-1])))
        win_m = np.zeros_like(conditions_m[0], dtype=bool)
        for condition in conditions_m:
             win_m |= condition
    # read fitted splines
    dfy = pd.read_csv(f"{outdir}/synth_Y_fit.csv")
    mu_y = np.array(dfy['0'])
    if args.d:
        # read fitted splines
        if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt":
            dfy_m = pd.read_csv(f"{outdir.rsplit('/',1)[0]}/synth_Y_fit.csv")
            mu_y_m = np.array(dfy_m['0'])
        elif protocol == "protocols/3_drug_protocol_23_10_24.mmt":
            dfy_m = pd.read_csv(f"outputs_real_16102024_MA1_FP_RT/{args.c}/synth_Y_fit.csv")
            mu_y_m = np.array(dfy_m['0'])
        elif protocol == "protocols/3_drug_protocol_14_11_24.mmt":
            dfy_m = pd.read_csv(f"outputs_real_30102024_MA_FP_RT/{args.c}/synth_Y_fit.csv")
            mu_y_m = np.array(dfy_m['0'])
    # fit model
    pars, sc = get_pars(model_num)
    print(f'{model_num}: {pars}, {sc}')
    print(args)
    if args.d:
        savedir = outdir + "/fits_milnes_and_opt"
    else:
        savedir = outdir + "/fits"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(f"{savedir}/{model_num}_fit_{len(win[(win == True)])}_points.csv", 'w') as f:
        f.write('"rep","pars","score"')
        f.write("\n")
        writer = csv.writer(f)
        writer.writerows(zip(np.arange(1, repeats+1, 1), pars, sc))
