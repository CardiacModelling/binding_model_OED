# import modules
import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from methods import steps, sd
import methods.funcs as funcs
import methods.parameters as parameters
import pints
import numpy as np
import pandas as pd
import argparse
import ast
import myokit
import csv
from scipy import special

parser = argparse.ArgumentParser(description='Plot fits and log-likelihoods')
parser.add_argument('-m', type=str, required=True, help='Model numbers')
parser.add_argument('-t', type=float, default=15e3, help='Max time')
parser.add_argument('-b', type=list, default=[[1e3, 11e3]], help='Protocol window(s) of interest')
parser.add_argument('-e', type=str, default='joey_sis', help='hERG model parameters')
parser.add_argument('-o', type=str, required=True, help='output folder for synthetic data')
parser.add_argument('-c', type=str, help='drug compound')
args = parser.parse_args()

def get_opt_prot(model_pars, herg, v_steps, t_steps, p0, CMAES_pop = 10, max_iter = 5, alt_protocol = None):

    class DrugBind(object):
        def __init__(self, p):
            self.pars = p

        def __call__(self, p):
            self.pars = p
            if alt_protocol is not None:
                self.v_st, self.t_st = funcs.get_steps(v_steps, t_steps, self.pars[:len(self.pars)//2])
                self.v_st_alt, self.t_st_alt = funcs.get_steps(v_steps, t_steps, self.pars[len(self.pars)//2:])
                prot = funcs.create_protocol(self.v_st, self.t_st)
                prot_alt = funcs.create_protocol(self.v_st_alt, self.t_st_alt)
            else:
                self.v_st, self.t_st = funcs.get_steps(v_steps, t_steps, self.pars)
                prot = funcs.create_protocol(self.v_st, self.t_st)
            if alt_protocol is not None:
                model_out, cont = funcs.model_outputs(model_pars, herg, prot, times = np.arange(0, np.sum(self.t_st), 10), concs = concs,
                                          alt_protocol = prot_alt, alt_times = np.arange(0, np.sum(self.t_st_alt), 10), wins = [1e3, np.sum(self.t_st[1:4])],
                                          wins_alt = [1e3, np.sum(self.t_st_alt[1:4])])
            else:
                model_out, cont = funcs.model_outputs(model_pars, herg, prot, times = np.arange(0, np.sum(self.t_st), 10), concs = concs, wins = [1e3, np.sum(self.t_st[1:4])])
            ### loop through models and concentrations to get traces
            all_traces = []
            for m in model_out:
                model_trace = []
                for c in concs:
                    model_trace = np.append(model_trace, model_out[m][c])
                all_traces.append(model_trace)
            conts = []
            for c in concs:
                conts = np.append(conts, cont)
            #ssq = [sum((m-n)**2) for i,m in enumerate(all_traces) for j,n in enumerate(all_traces) if i < j]
            lhoods = []
            for i,m in enumerate(all_traces):
                for j,n in enumerate(all_traces):
                    if i < j:
                        top = np.exp(-conts**2*((m**2+1)/(2*sd**2))) + np.sqrt(np.pi/2)*(conts/sd)*np.sqrt(1+m**2)*special.erf((conts/(np.sqrt(2)*sd))*np.sqrt(1+m**2))
                        bottom = np.exp(-conts**2*((n**2+1)/(2*sd**2))) + np.sqrt(np.pi/2)*(conts/sd)*((1+m*n)/np.sqrt(1+m**2))*special.erf((conts*(1+m*n))/(np.sqrt(2)*sd*np.sqrt(1+m**2)))*np.exp(-(conts**2*(m-n)**2)/(2*sd**2*(1+m**2)))
                        lhoods.append(-2*np.sum(np.log(top/bottom)))
            #out = -np.median(ssq)
            out = np.median(lhoods)
            return out

        def n_parameters(self):
            return len(self.pars)

    lower_v = [-50]*v_steps.count(np.nan)
    upper_v = [40]*v_steps.count(np.nan)
    lower_t_p = [1000]*(t_steps.count(np.nan)-1)
    upper_t_p = [5000]*(t_steps.count(np.nan)-1)
    lower_t_i = [50]
    upper_t_i = [20000]
    step_v = [5]*v_steps.count(np.nan)
    step_t = [50]*t_steps.count(np.nan)

    if alt_protocol is not None:
        boundaries = pints.RectangularBoundaries(lower_v + lower_t_p + lower_t_i + lower_v + lower_t_p + lower_t_i,
                                                 upper_v + upper_t_p + upper_t_i + upper_v + upper_t_p + upper_t_i)
    else:
        boundaries = pints.RectangularBoundaries(lower_v + lower_t_p + lower_t_i, upper_v + upper_t_p + upper_t_i)
    transformation = pints.RectangularBoundariesTransformation(boundaries)

    if alt_protocol is not None:
        design = DrugBind(p0+alt_protocol)
    else:
        design = DrugBind(p0)
    score = 1e15

    # Fix random seed for reproducibility
    np.random.seed(101)

    for i in range(0,100):
        q0 = boundaries.sample()[0]
        try:
            temp = design(q0)
            if temp < score:
                score = temp
                q0save = q0
        except Exception as e:
            print(f"An error occurred: {e}")

    optimiser = pints.CMAES
    if alt_protocol is not None:
        design = DrugBind(p0+alt_protocol)
        opt = pints.OptimisationController(
            design,
            q0save,
            sigma0=step_v+step_t+step_v+step_t,
            transformation=transformation,
            method=optimiser)
    else:
        design = DrugBind(p0)
        opt = pints.OptimisationController(
            design,
            q0save,
            sigma0=step_v+step_t,
            transformation=transformation,
            method=optimiser)

    opt.optimiser().set_population_size(CMAES_pop)
    opt.set_max_iterations(max_iter)
    opt.set_max_unchanged_iterations(iterations=20, threshold=1e-2)
    opt.set_parallel(-1)

    try:
        # Tell numpy not to issue warnings
        with np.errstate(all='ignore'):
            p, s = opt.run()
            return p, s
    except ValueError:
        import traceback
        traceback.print_exc()
        raise RuntimeError('Not here...')

def main(model_nums, max_time, bounds, herg, output_folder):
    v_steps = [-80, np.nan, np.nan, np.nan, -80]
    t_steps = [1000, np.nan, np.nan, np.nan, np.nan]
    p0_1 = [0, 0, 0, 3340, 3330, 3330, 14000]
    p0_2 = [0, 0, 0, 3340, 3330, 3330, 14000]

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
    for j, m in enumerate(model_nums):
        df = pd.read_csv(f'{output_folder}/fits/{model_nums[j]}_fit_{length}_points.csv')
        parstring = df.loc[df['score'].idxmax()]['pars']
        cleaned_string = parstring.replace("[", "").replace("]", "").replace("\n", "").strip()
        parlist = [float(i) for i in cleaned_string.split(" ") if i]
        drug_fit_pars[m] = parlist

    # perform optimisation
    p_out, cost = get_opt_prot(drug_fit_pars, herg, v_steps, t_steps, p0_1, CMAES_pop = 7, max_iter = 240, alt_protocol = p0_2)
    print(f'Final objective cost: {cost}')
    print(f'Final optimised params: {p_out}')

    # generate and save optimised protocol
    vsteps_all = [-80, np.nan, np.nan, np.nan, -80, -80, np.nan, np.nan, np.nan, -80]
    tsteps_all = [1000, np.nan, np.nan, np.nan, np.nan, 1000, np.nan, np.nan, np.nan, np.nan]
    p_ordered = list(p_out[:3]) + list(p_out[7:10]) + list(p_out[3:7]) + list(p_out[10:])
    vst,tst = funcs.get_steps(vsteps_all, tsteps_all, p_ordered)
    prot_full = funcs.create_protocol(vst, tst)
    myokit.save(filename = f'{output_folder}/opt_prot.mmt', protocol = prot_full)

    # save length and alternate window
    full_dur = sum(tst)
    win = [1000, sum(tst[:4])]
    alt_win = [sum(tst[:6]), sum(tst[:-1])]
    t_steps = [tst[1], tst[2], tst[3], tst[6], tst[7], tst[8]]

    # write to CSV
    with open(f'{output_folder}/prot_details.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(win)
        writer.writerow(alt_win)
        writer.writerow([full_dur])
        writer.writerow(t_steps)

if __name__ == "__main__":
    concs = parameters.drug_concs[args.c]
    m_list = ast.literal_eval(args.m)
    main(m_list, args.t, args.b, args.e, args.o)
