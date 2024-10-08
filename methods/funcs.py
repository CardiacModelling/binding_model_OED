import pandas as pd
import numpy as np
from . import herg_pars, sweeps, steps
from .models import Model
import myokit

### Function for generating synthetic data
def generate_data(herg_model, drug_vals, prot, sd, max_time, bounds, m_sel, concs):
    # get herg parameters
    if herg_model == '2019_37C':
        herg_vals = [2.07e-3, 7.17e-2, 3.44e-5, 6.18e-2, 4.18e-1, 2.58e-2, 4.75e-2, 2.51e-2, 33.3]
    elif herg_model == 'kemp':
        herg_vals = []

    if max_time != 15e3:
        swps = 5
    else:
        swps = sweeps

    # define protocol
    protocol = myokit.load_protocol(prot)

    # define simulation time
    times = np.arange(0, max_time, steps)
    wins = []
    for b in bounds:
        wins.append((times >= b[0]) & (times < b[-1]))

    # initialise synth data dict
    synth_X_win = {}
    synth_X = {}

    # set random seed
    np.random.seed(10)

    # loop through concs
    for conc in concs:
        # initialise list for data
        X = []
        Y = []
        X_full = []
        Y_full = []

        # load model
        if herg_model != 'kemp':
            model = Model(f'm{m_sel}',
                            protocol,
                            parameters=['binding'],
                            analytical=True)

            # set hERG model parameters
            model.set_fix_parameters({p:v for p, v in zip(herg_pars, herg_vals)})
        else:
            model = Model(f'kemp-m{m_sel}',
                            protocol,
                            parameters=['binding'],
                            analytical=True)

        # fix kt if necessary
        if m_sel in ['12', '13']:
            model.fix_kt()

        # run control
        model.set_dose(0)
        control = model.simulate(drug_vals, times)
        if sd != 0:
            control = control + np.random.normal(0,sd,len(control))

        # run drug sweep
        model.set_dose(conc)
        after = model.simulate(drug_vals, times)
        if sd != 0:
            after = after + np.random.normal(0,sd,len(after))

        for win in wins:
            Y = np.append(Y, control[win])
            X = np.append(X, after[win])

        Y_full = np.append(Y_full, control)
        X_full = np.append(X_full, after)

        # run multiple sweeps
        for i in range(1, swps):
            after = model.simulate(drug_vals, times, reset = False)
            if sd != 0:
                after = after + np.random.normal(0,sd,len(after))
            for win in wins:
                X = np.append(X, after[win])
                Y = np.append(Y, control[win])
            Y_full = np.append(Y_full, control)
            X_full = np.append(X_full, after)

        synth_X_win[conc] = X
        synth_Y_win = Y
        synth_X[conc] = X_full
        synth_Y = Y_full

    synth_Z = {}
    synth_Z_win = {}
    for conc in concs:
        synth_Z[conc] = synth_X[conc]/synth_Y
        synth_Z_win[conc] = synth_X_win[conc]/synth_Y_win

    ts = np.arange(0, len(synth_Y)*steps, steps)
    ts_win = np.arange(0, len(synth_Y_win)*steps, steps)

    return synth_X, synth_Y, synth_Z, synth_X_win, synth_Y_win, synth_Z_win, ts, ts_win

### Function for extracting protocol steps
def get_steps(v_steps, t_steps, pars):
    v_st = np.copy(v_steps)
    t_st = np.copy(t_steps)
    v_nan = (i for i, v in enumerate(v_steps) if np.isnan(v))
    t_nan = (i for i, t in enumerate(t_steps) if np.isnan(t))
    for p_i in pars:
        try:
            v_ind = next(v_nan)
            v_st[v_ind] = p_i
        except:
            try:
                t_ind = next(t_nan)
                t_st[t_ind] = p_i
            except:
                print("mismatch between voltages/steps to optimise and p0")
    return v_st, t_st

### Function for creating protocols
def create_protocol(v_steps, t_steps):
    if len(v_steps) != len(t_steps):
        print("Invalid protocol")
        return None
    else:
        prot = myokit.Protocol()
        for v,t in zip(v_steps, t_steps):
            prot.add_step(v, t)
    return prot

### Function for getting model outputs
def model_outputs(model_pars, herg_model, prot, times, concs, alt_protocol = None, alt_times = None, wins = [1e3, 11e3]):
    # get herg parameters
    if herg_model == '2019_37C':
        herg_vals = [2.07e-3, 7.17e-2, 3.44e-5, 6.18e-2, 4.18e-1, 2.58e-2, 4.75e-2, 2.51e-2, 33.3]
    elif herg_model == 'kemp':
        herg_vals = []
    model_out = {}
    swps = 5 #when generating optimal protocol, we use 5 sweeps
    win = (times >= wins[0]) & (times < wins[1])
    if alt_protocol is not None:
        win_alt = (alt_times >= wins[0]) & (alt_times < wins[1])
    for m in model_pars:
        binding_params = model_pars[m]
        model_out[m] = {}
        for conc in concs:
            if herg_model != 'kemp':
                model = Model(f'm{m}',
                                prot,
                                parameters=['binding'],
                                analytical=True)
                # set hERG model parameters
                model.set_fix_parameters({p:v for p, v in zip(herg_pars, herg_vals)})
            else:
                model = Model(f'kemp-m{m}',
                                prot,
                                parameters=['binding'],
                                analytical=True)
            # fix kt if necessary
            if m in ['12', '13']:
                model.fix_kt()
            try:
                model_milnes = []
                model.set_dose(conc)
                after = model.simulate(binding_params, times)[win]
                model_milnes = np.append(model_milnes, after)
                for i in range(swps-1):
                    if (alt_protocol is not None) & ((i % 2) == 0):
                        model.change_protocol(alt_protocol)
                        after = model.simulate(binding_params, alt_times, reset=False)[win_alt]
                        model_milnes = np.append(model_milnes, after)
                        model.change_protocol(prot)
                    else:
                        after = model.simulate(binding_params, times, reset=False)[win]
                        model_milnes = np.append(model_milnes, after)
                model_out[m][conc] = model_milnes
            except:
                model_out[m][conc] = np.ones(times.shape) * float('inf')
    return model_out
