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
parser.add_argument('-e', action='store_true', help='Enable fitting of 2x optimal protocol data')
parser.add_argument('-a', action='store_true', help='Enable fitting of Milnes and 2x optimal protocol data')
args = parser.parse_args()

def approx_hessian_logspace(f, phi_hat, h=1e-5):
    """
    Approximate the Hessian via finite differences.

    Parameters:
        f : log-likelihood function
        phi_hat : log-transformed MLE
        h : finite diff step size (in log-space)

    Returns:
        H : approx hessian
    """
    phi_hat = np.asarray(phi_hat)
    n = phi_hat.size
    H = np.zeros((n, n))

    def g(phi_):
        theta = np.exp(phi_)  # Convert back to original space
        return f(theta)

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = 1
            ej[j] = 1

            if i == j:
                fpp = g(phi_hat + h * ei)
                fmm = g(phi_hat - h * ei)
                f0 = g(phi_hat)
                H[i, i] = (fpp - 2 * f0 + fmm) / h**2
            else:
                fpp = g(phi_hat + h * ei + h * ej)
                fpm = g(phi_hat + h * ei - h * ej)
                fmp = g(phi_hat - h * ei + h * ej)
                fmm = g(phi_hat - h * ei - h * ej)
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h**2)

    return H

def get_pars(model_num):
    ### Set individual errors and weights
    likelihoods = []
    for conc in concs:
        # Create forward models
        if herg_model != 'kemp' and herg_model != '2024_Joey_sis_25C' and herg_model != 'temp_dep':
            model = classes.ConcatMilnesModel(f'm{model_num}', protocol, times,
                                          win, conc, param_dict)
            if args.d:
                if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt" and protocol != "protocols/3_drug_protocol_28_11_24.mmt":
                    model_m = classes.ConcatMilnesModel(f'm{model_num}', 'protocols/Milnes_Phil_Trans.mmt', times_m,
                                          win_m, conc, param_dict)
                else:
                    model_m = classes.ConcatMilnesModel(f'm{model_num}', 'protocols/Milnes_16102024_MA1_FP_RT.mmt', times_m,
                                          win_m, conc, param_dict)
        elif herg_model == 'temp_dep':
            model = classes.ConcatMilnesModel(f'm{model_num}-td', protocol, times,
                                          win, conc, param_dict)
            if args.d:
                model = classes.ConcatMilnesModel(f'm{model_num}-td', protocol, times,
                                          win, conc, param_dict, additional_pars=3)
                model_m = classes.ConcatMilnesModel(f'm{model_num}-td', 'protocols/Milnes_16102024_MA1_FP_RT.mmt', times_m,
                                          win_m, conc, param_dict, multi=args.d)
            elif args.a:
                model = classes.ConcatMilnesModel(f'm{model_num}-td', protocol, times,
                                          win, conc, param_dict, additional_pars=6)
                model_m = classes.ConcatMilnesModel(f'm{model_num}-td', 'protocols/Milnes_16102024_MA1_FP_RT.mmt', times_m,
                                          win_m, conc, param_dict, additional_pars=3, multi=True)
                model_o = classes.ConcatMilnesModel(f'm{model_num}-td', 'protocols/3_drug_protocol_14_11_24.mmt', times_o,
                                          win_o, conc, param_dict, all=True)
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
                if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt" and protocol != "protocols/3_drug_protocol_28_11_24.mmt":
                    model_m = classes.ConcatMilnesModel(f'sis-m{model_num}', 'protocols/Milnes_Phil_Trans.mmt', times_m,
                                          win_m, conc, param_dict)
                else:
                    model = classes.ConcatMilnesModel(f'sis-m{model_num}', protocol, times,
                                          win, conc, param_dict, additional_pars=3)
                    model_m = classes.ConcatMilnesModel(f'sis-m{model_num}', 'protocols/Milnes_16102024_MA1_FP_RT.mmt', times_m,
                                          win_m, conc, param_dict, multi=args.d)
            if args.e:
                model_m = classes.ConcatMilnesModel(f'sis-m{model_num}', 'protocols/3_drug_protocol_14_11_24.mmt', times_m,
                                          win_m, conc, param_dict)
            if args.a:
                model_m = classes.ConcatMilnesModel(f'sis-m{model_num}', 'protocols/Milnes_16102024_MA1_FP_RT.mmt', times_m,
                                          win_m, conc, param_dict)
                model_o = classes.ConcatMilnesModel(f'sis-m{model_num}', 'protocols/3_drug_protocol_14_11_24.mmt', times_o,
                                          win_o, conc, param_dict)
        # Load data
        u = np.loadtxt(
            f'{outdir}/fb_synthetic_conc_{conc}.csv',
            delimiter=',',
            skiprows=1
        )
        concat_time = u[:, 0]
        concat_milnes = u[:, 1]
        if args.d:
            if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt" and protocol != "protocols/3_drug_protocol_28_11_24.mmt":
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
            elif protocol == "protocols/3_drug_protocol_14_11_24.mmt" or protocol == "protocols/3_drug_protocol_28_11_24.mmt":
                if herg_model == 'temp_dep' or herg_model == '2024_Joey_sis_25C':
                    # Load data
                    u_m = np.loadtxt(
                        f'outputs_real_30102024_MA_FP_RT_td_sis/{args.c}/fb_synthetic_conc_{conc}.csv',
                        delimiter=',',
                        skiprows=1
                    )
                else:
                    # Load data
                    u_m = np.loadtxt(
                        f'outputs_real_30102024_MA_FP_RT/{args.c}/fb_synthetic_conc_{conc}.csv',
                        delimiter=',',
                        skiprows=1
                    )
            concat_time_m = u_m[:, 0]
            concat_milnes_m = u_m[:, 1]
        if args.e:
            # Load data
            u_m = np.loadtxt(
                f'outputs_real_20241114_MA_FP_RT/{args.c}/fb_synthetic_conc_{conc}.csv',
                delimiter=',',
                skiprows=1
            )
            concat_time_m = u_m[:, 0]
            concat_milnes_m = u_m[:, 1]
        if args.a:
            if herg_model == 'temp_dep':
                # Load data
                u_m = np.loadtxt(
                    f'outputs_real_30102024_MA_FP_RT_td/{args.c}/fb_synthetic_conc_{conc}.csv',
                    delimiter=',',
                    skiprows=1
                )
                # Load data
                u_o = np.loadtxt(
                    f'outputs_real_20241114_MA_FP_RT_td/{args.c}/fb_synthetic_conc_{conc}.csv',
                    delimiter=',',
                    skiprows=1
                )
            else:
                # Load data
                u_m = np.loadtxt(
                    f'outputs_real_30102024_MA_FP_RT/{args.c}/fb_synthetic_conc_{conc}.csv',
                    delimiter=',',
                    skiprows=1
                )
                # Load data
                u_o = np.loadtxt(
                    f'outputs_real_20241114_MA_FP_RT/{args.c}/fb_synthetic_conc_{conc}.csv',
                    delimiter=',',
                    skiprows=1
                )
            concat_time_m = u_m[:, 0]
            concat_milnes_m = u_m[:, 1]
            concat_time_o = u_o[:, 0]
            concat_milnes_o = u_o[:, 1]
        # Create single output problem
        problem = pints.SingleOutputProblem(model, concat_time, concat_milnes)
        if herg_model == 'temp_dep': # or herg_model == '2024_Joey_sis_25C':
            if args.a:
                likelihoods.append(classes.NormalRatioLogLikelihood(problem, mu_y, td=True, all=True))
            else:
                likelihoods.append(classes.NormalRatioLogLikelihood(problem, mu_y, td=True, multi=args.d))
        else:
            likelihoods.append(classes.NormalRatioLogLikelihood(problem, mu_y, conc = conc))
        if args.d or args.e or args.a:
            problem_m = pints.SingleOutputProblem(model_m, concat_time_m, concat_milnes_m)
            if herg_model == 'temp_dep' or herg_model == '2024_Joey_sis_25C':
                if args.a:
                    likelihoods.append(classes.NormalRatioLogLikelihood(problem_m, mu_y_m, td=True, all=True))
                else:
                    likelihoods.append(classes.NormalRatioLogLikelihood(problem_m, mu_y_m, td=True, multi=True))
            else:
                likelihoods.append(classes.NormalRatioLogLikelihood(problem_m, mu_y_m))
        if args.a:
            problem_o = pints.SingleOutputProblem(model_o, concat_time_o, concat_milnes_o)
            if herg_model == 'temp_dep':
                likelihoods.append(classes.NormalRatioLogLikelihood(problem_o, mu_y_o, td=True, all=True))
            else:
                likelihoods.append(classes.NormalRatioLogLikelihood(problem_o, mu_y_o))

    if len(likelihoods) > 1:
        f = pints.SumOfIndependentLogPDFs(likelihoods)
    else:
        f = likelihoods[0]
    if herg_model == 'temp_dep': # or herg_model == '2024_Joey_sis_25C':
        bounds = boundaries.Boundaries(model_num, fix_hill=False, likelihood=True, td=True, multi=args.d, all=args.a)
    else:
        bounds = boundaries.Boundaries(model_num, fix_hill=False, likelihood=True)

    # Fix random seed for reproducibility
    np.random.seed(101)
    # Transformation
    if herg_model == 'temp_dep': # or herg_model == '2024_Joey_sis_25C':
        if args.d:
            if model_num in ['12', '13']:
                transform = pints.ComposedTransformation(
                    pints.LogTransformation(f.n_parameters() - 8),
                    pints.IdentityTransformation(3),
                    pints.LogTransformation(1),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(2),
                )
            else:
                transform = pints.ComposedTransformation(
                    pints.LogTransformation(f.n_parameters() - 7),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(1),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(2),
                )
        elif args.a:
            if model_num in ['12', '13']:
                transform = pints.ComposedTransformation(
                    pints.LogTransformation(f.n_parameters() - 11),
                    pints.IdentityTransformation(3),
                    pints.LogTransformation(1),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(1),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(2),
                )
            else:
                transform = pints.ComposedTransformation(
                    pints.LogTransformation(f.n_parameters() - 10),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(1),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(1),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(2),
                )

        else:
            if model_num in ['12', '13']:
                transform = pints.ComposedTransformation(
                    pints.LogTransformation(f.n_parameters() - 5),
                    pints.IdentityTransformation(3),
                    pints.LogTransformation(2),
                )
            else:
                transform = pints.ComposedTransformation(
                    pints.LogTransformation(f.n_parameters() - 4),
                    pints.IdentityTransformation(2),
                    pints.LogTransformation(2),
                )
    else:
        if model_num in ['12', '13']:
            transform = pints.ComposedTransformation(
                pints.LogTransformation(f.n_parameters() - 2),
                pints.IdentityTransformation(1),
                pints.LogTransformation(1),
            )
        else:
            transform = pints.LogTransformation(f.n_parameters())
    params, scores, hessians = [], [], []
    for i in range(repeats):
        q0 = bounds.sample()
        while math.isnan(f(q0)) or math.isinf(f(q0)):
            q0 = bounds.sample()
            print(f'resample evaluation: {f(q0)}')
        # Create optimiser
        print('into optimiser')
        print(f'q0: {q0}')
        print(f'f(q0): {f(q0)}')
        opt = pints.OptimisationController(
                f, q0, boundaries=bounds, transformation=transform, method=pints.CMAES)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(iterations=120, threshold=1e-2)  #200, 1e-3
        # Run optimisation
        try:
            with np.errstate(all='ignore'): # Tell numpy not to issue warnings
                p, s = opt.run()
                params.append(p)
                scores.append(s)
                hessians.append(approx_hessian_logspace(f, np.log(p)))
        except ValueError:
            import traceback
            traceback.print_exc()
    return params, scores, hessians

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
    if herg_model != '2024_Joey_sis_25C' and herg_model != 'temp_dep' and herg_model != '2025_Frankie_staircase_25C':
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
        concs = [0]
    if herg_model != 'kemp' and herg_model != '2024_Joey_sis_25C' and herg_model != 'temp_dep':
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
        if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt" and protocol != "protocols/3_drug_protocol_28_11_24.mmt":
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
    if args.e:
        times_m = np.arange(0, 12.3e3, steps)
        conditions_m = []
        for b in [[1000,3900],[5200,8300]]:
            conditions_m.append(((times_m >= b[0]) & (times_m < b[-1])))
        win_m = np.zeros_like(conditions_m[0], dtype=bool)
        for condition in conditions_m:
             win_m |= condition
    if args.a:
        times_m = np.arange(0, 15.35e3, steps)
        conditions_m = []
        for b in [[1.35e3, 11.35e3]]:
            conditions_m.append(((times_m >= b[0]) & (times_m < b[-1])))
        times_o = np.arange(0, 12.3e3, steps)
        conditions_o = []
        for b in [[1500,3900],[5600,8200]]:
            conditions_o.append(((times_o >= b[0]) & (times_o < b[-1])))
        win_m = np.zeros_like(conditions_m[0], dtype=bool)
        win_o = np.zeros_like(conditions_o[0], dtype=bool)
        for condition in conditions_m:
             win_m |= condition
        for condition in conditions_o:
             win_o |= condition

    # read fitted splines
    dfy = pd.read_csv(f"{outdir}/synth_Y_fit.csv")
    mu_y = np.array(dfy['0'])
    if args.d:
        # read fitted splines
        if protocol != "protocols/3_drug_protocol_23_10_24.mmt" and protocol != "protocols/3_drug_protocol_14_11_24.mmt" and protocol != "protocols/3_drug_protocol_28_11_24.mmt":
            dfy_m = pd.read_csv(f"{outdir.rsplit('/',1)[0]}/synth_Y_fit.csv")
            mu_y_m = np.array(dfy_m['0'])
        elif protocol == "protocols/3_drug_protocol_23_10_24.mmt":
            dfy_m = pd.read_csv(f"outputs_real_16102024_MA1_FP_RT/{args.c}/synth_Y_fit.csv")
            mu_y_m = np.array(dfy_m['0'])
        elif protocol == "protocols/3_drug_protocol_14_11_24.mmt" or protocol == "protocols/3_drug_protocol_28_11_24.mmt":
            if herg_model == 'temp_dep' or herg_model == '2024_Joey_sis_25C':
                dfy_m = pd.read_csv(f"outputs_real_30102024_MA_FP_RT_td_sis/{args.c}/synth_Y_fit.csv")
                mu_y_m = np.array(dfy_m['0'])
            else:
                dfy_m = pd.read_csv(f"outputs_real_30102024_MA_FP_RT/{args.c}/synth_Y_fit.csv")
                mu_y_m = np.array(dfy_m['0'])
    if args.e:
        dfy_m = pd.read_csv(f"outputs_real_20241114_MA_FP_RT/{args.c}/synth_Y_fit.csv")
        mu_y_m = np.array(dfy_m['0'])
    if args.a:
        if herg_model == 'temp_dep':
            dfy_m = pd.read_csv(f"outputs_real_30102024_MA_FP_RT_td/{args.c}/synth_Y_fit.csv")
            dfy_o = pd.read_csv(f"outputs_real_20241114_MA_FP_RT_td/{args.c}/synth_Y_fit.csv")
        else:
            dfy_m = pd.read_csv(f"outputs_real_30102024_MA_FP_RT/{args.c}/synth_Y_fit.csv")
            dfy_o = pd.read_csv(f"outputs_real_20241114_MA_FP_RT/{args.c}/synth_Y_fit.csv")
        mu_y_m = np.array(dfy_m['0'])
        mu_y_o = np.array(dfy_o['0'])
    # fit model
    pars, sc, hessians = get_pars(model_num)
    print(f'{model_num}: {pars}, {sc}, {hessians}')
    print(args)
    if args.d:
        savedir = outdir + "/fits_milnes_and_opt"
    elif args.e:
        savedir = outdir + "/fits_two_opt"
    elif args.a:
        savedir = outdir + "/fits_allthree"
    else:
        savedir = outdir + "/fits"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(f"{savedir}/{model_num}_fit_{len(win[(win == True)])}_points.csv", 'w') as f:
        f.write('"rep","pars","score"')
        f.write("\n")
        writer = csv.writer(f)
        writer.writerows(zip(np.arange(1, repeats+1, 1), pars, sc))
    with open(f"{savedir}/{model_num}_hess_{len(win[(win == True)])}_points.csv", 'w') as f:
        f.write('"rep","hessian"')
        f.write("\n")
        writer = csv.writer(f)
        writer.writerows(zip(np.arange(1, repeats+1, 1), hessians))
