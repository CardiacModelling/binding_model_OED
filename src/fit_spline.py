import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from methods import sweeps
import pandas as pd
import numpy as np
from scipy.linalg import solve
import skfda
import argparse
import ast

parser = argparse.ArgumentParser(description='Spline fitting')
parser.add_argument('-i', type=str, required=True, help='Input folder')
parser.add_argument('-o', type=str, required=True, help='Output folder')
parser.add_argument('-l', type=float, default=5, help='Smoothing parameter lambda')
parser.add_argument('-s', type=str, default="[10000]", help="List of integers for the length of each voltage step e.g. '[1,2,3,4,5]'")
parser.add_argument('-m', type=str, default='milnes', help='Milnes, opt, or real')
parser.add_argument('-t', type=float, default=15e3, help='Max time')
args = parser.parse_args()

def main(input, output, lambda_, t_steps, data, max_t):
    df_all = pd.read_csv(f"{input}/synth_Y.csv")
    # Parameters (TODO currently hardcoded to get no. of swps)
    if data == 'milnes':
        t_swp = 10000
        swps = sweeps
    elif data == 'opt':
        j = 0
        t_swp = t_steps[j]
        t_total = 0
        swps = int(np.floor(250000/max_t))*6
    elif data == 'milnes real':
        t_swp = 10000
        swps = 10
    elif data == '24102024_MA_FP_RT':
        t_swp_1 = 10053
        t_swp_2 = 14997
        t_total = t_swp_1 + t_swp_2
        swps = 13
    elif data == '20241114_MA_FP_RT':
        t_swp_1 = 400
        t_swp_2 = 100
        t_swp_3 = 2400
        t_swp_4 = 400
        t_swp_5 = 2600
        t_swp_6 = 100
        t_total = t_swp_1 + t_swp_2 + t_swp_3 + t_swp_4 + t_swp_5 + t_swp_6
        swps = 10
    elif data == '20241128_MA_FP_RT':
        t_swp_1 = 897
        t_swp_2 = 691
        t_swp_3 = 2066
        t_total = t_swp_1 + t_swp_2 + t_swp_3
        swps = 10
    elif data == '20241128_MA_FP_RT_2':
        t_swp_1 = 100
        t_swp_2 = 1100
        t_total = t_swp_1*13 + t_swp_2
        swps = 9
    else:
        lens = [3340, 3330, 3340, 3330, 3330]
        t_total = 0
        swps = sweeps*5
    n_order = 4
    all_ = []

    # Fit splines
    for i in range(swps):
        if data == 'milnes' or data == 'milnes real':
            df_rep = df_all[(df_all['t'] >= t_swp * i) & (df_all['t'] < t_swp * (i + 1))][['t', 'x']]
            knots = np.arange(t_swp * i, t_swp * (i + 1) + t_swp/100, t_swp/100)
        elif data == 'opt':
            df_rep = df_all[(df_all['t'] >= t_total) & (df_all['t'] < t_total + t_swp)][['t', 'x']]
            knots = np.arange(t_total, t_total + t_swp, t_swp/2)
            if j == len(t_steps)-1:
                j=0
            else:
                j+=1
            t_total += t_swp
            t_swp = t_steps[j]
        elif data == '24102024_MA_FP_RT':
            df_rep_1 = df_all[(df_all['t'] >= t_total * i) & (df_all['t'] < t_total * (i + 1) - t_swp_2)][['t', 'x']]
            df_rep_2 = df_all[(df_all['t'] >= t_total * i + t_swp_1) & (df_all['t'] < t_total * (i + 1))][['t', 'x']]
            knots_1 = np.arange(t_total * i, t_total * (i + 1) - t_swp_2 + t_swp_1/2, t_swp_1/2)
            knots_2 = np.arange(t_total * i + t_swp_1, t_total * (i + 1) + t_swp_2/2, t_swp_2/2)
        elif data == '20241114_MA_FP_RT':
            df_rep_1 = df_all[(df_all['t'] >= t_total * i) & (df_all['t'] < t_total * i + t_swp_1)][['t', 'x']]
            df_rep_2 = df_all[(df_all['t'] >= t_total * i + t_swp_1) & (df_all['t'] < t_total * i + t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_3 = df_all[(df_all['t'] >= t_total * i + t_swp_1 + t_swp_2) & (df_all['t'] < t_total * i + t_swp_1 + t_swp_2 + t_swp_3)][['t', 'x']]
            df_rep_4 = df_all[(df_all['t'] >= t_total * i + t_swp_1 + t_swp_2 + t_swp_3) & (df_all['t'] < t_total * i + t_swp_1 + t_swp_2 + t_swp_3 + t_swp_4)][['t', 'x']]
            df_rep_5 = df_all[(df_all['t'] >= t_total * i + t_swp_1 + t_swp_2 + t_swp_3 + t_swp_4) & (df_all['t'] < t_total * i + t_total - t_swp_6)][['t', 'x']]
            df_rep_6 = df_all[(df_all['t'] >= t_total * i + t_total - t_swp_6) & (df_all['t'] < t_total * i + t_total)][['t', 'x']]
            knots_1 = np.arange(t_total * i, t_total * i + t_swp_1 + t_swp_1/2, t_swp_1/2)
            knots_2 = np.arange(t_total * i + t_swp_1, t_total * i + t_swp_1 + t_swp_2 + t_swp_2/2, t_swp_2/2)
            knots_3 = np.arange(t_total * i + t_swp_1 + t_swp_2, t_total * i + t_swp_1 + t_swp_2 + t_swp_3 + t_swp_3/20, t_swp_3/20)
            knots_4 = np.arange(t_total * i + t_swp_1 + t_swp_2 + t_swp_3, t_total * i + t_swp_1 + t_swp_2 + t_swp_3 + t_swp_4 + t_swp_4/2, t_swp_4/2)
            knots_5 = np.arange(t_total * i + t_swp_1 + t_swp_2 + t_swp_3 + t_swp_4, t_total * i + t_total - t_swp_6 + t_swp_5/20, t_swp_5/20)
            knots_6 = np.arange(t_total * i + t_total - t_swp_6, t_total * i + t_total + t_swp_6/2, t_swp_6/2)
        elif data == '20241128_MA_FP_RT':
            df_rep_1 = df_all[(df_all['t'] >= t_total * i) & (df_all['t'] < t_total * i + t_swp_1)][['t', 'x']]
            df_rep_2 = df_all[(df_all['t'] >= t_total * i + t_swp_1) & (df_all['t'] < t_total * i + t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_3 = df_all[(df_all['t'] >= t_total * i + t_total - t_swp_3) & (df_all['t'] < t_total * i + t_total)][['t', 'x']]
            knots_1 = np.arange(t_total * i, t_total * i + t_swp_1 + t_swp_1/4, t_swp_1/4)
            knots_2 = np.arange(t_total * i + t_swp_1, t_total * i + t_swp_1 + t_swp_2 + t_swp_2/4, t_swp_2/4)
            knots_3 = np.arange(t_total * i + t_total - t_swp_3, t_total * i + t_total + t_swp_3/10, t_swp_3/10)
        elif data == '20241128_MA_FP_RT_2':
            df_rep_1 = df_all[(df_all['t'] >= t_total * i) & (df_all['t'] < t_total * i + t_swp_1)][['t', 'x']]
            df_rep_2 = df_all[(df_all['t'] >= t_total * i + t_swp_1) & (df_all['t'] < t_total * i + 2 * t_swp_1)][['t', 'x']]
            df_rep_3 = df_all[(df_all['t'] >= t_total * i + 2 * t_swp_1) & (df_all['t'] < t_total * i + 3 * t_swp_1)][['t', 'x']]
            df_rep_4 = df_all[(df_all['t'] >= t_total * i + 3 * t_swp_1) & (df_all['t'] < t_total * i + 4 * t_swp_1)][['t', 'x']]
            df_rep_5 = df_all[(df_all['t'] >= t_total * i + 4 * t_swp_1) & (df_all['t'] < t_total * i + 5 * t_swp_1)][['t', 'x']]
            df_rep_6 = df_all[(df_all['t'] >= t_total * i + 5 * t_swp_1) & (df_all['t'] < t_total * i + 6 * t_swp_1)][['t', 'x']]
            df_rep_7 = df_all[(df_all['t'] >= t_total * i + 6 * t_swp_1) & (df_all['t'] < t_total * i + 7 * t_swp_1)][['t', 'x']]
            df_rep_8 = df_all[(df_all['t'] >= t_total * i + 7 * t_swp_1) & (df_all['t'] < t_total * i + 7 * t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_9 = df_all[(df_all['t'] >= t_total * i + 7 * t_swp_1 + t_swp_2) & (df_all['t'] < t_total * i + 8 * t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_10 = df_all[(df_all['t'] >= t_total * i + 8 * t_swp_1 + t_swp_2) & (df_all['t'] < t_total * i + 9 * t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_11 = df_all[(df_all['t'] >= t_total * i + 9 * t_swp_1 + t_swp_2) & (df_all['t'] < t_total * i + 10 * t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_12 = df_all[(df_all['t'] >= t_total * i + 10 * t_swp_1 + t_swp_2) & (df_all['t'] < t_total * i + 11 * t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_13 = df_all[(df_all['t'] >= t_total * i + 11 * t_swp_1 + t_swp_2) & (df_all['t'] < t_total * i + 12 * t_swp_1 + t_swp_2)][['t', 'x']]
            df_rep_14 = df_all[(df_all['t'] >= t_total * i + 12 * t_swp_1 + t_swp_2) & (df_all['t'] < t_total * i + 13 * t_swp_1 + t_swp_2)][['t', 'x']]
            knots_1 = np.arange(t_total * i, t_total * i + t_swp_1 + t_swp_1/3, t_swp_1/3)
            knots_2 = np.arange(t_total * i + t_swp_1, t_total * i + 2 * t_swp_1 + t_swp_1/3, t_swp_1/3)
            knots_3 = np.arange(t_total * i + 2 * t_swp_1, t_total * i + 3 * t_swp_1 + t_swp_1/3, t_swp_1/3)
            knots_4 = np.arange(t_total * i + 3 * t_swp_1, t_total * i + 4 * t_swp_1 + t_swp_1/3, t_swp_1/3)
            knots_5 = np.arange(t_total * i + 4 * t_swp_1, t_total * i + 5 * t_swp_1 + t_swp_1/3, t_swp_1/3)
            knots_6 = np.arange(t_total * i + 5 * t_swp_1, t_total * i + 6 * t_swp_1 + t_swp_1/3, t_swp_1/3)
            knots_7 = np.arange(t_total * i + 6 * t_swp_1, t_total * i + 7 * t_swp_1 + t_swp_1/3, t_swp_1/3)
            knots_8 = np.arange(t_total * i + 7 * t_swp_1, t_total * i + 7 * t_swp_1 + t_swp_2 + t_swp_2/5, t_swp_2/5)
            knots_9 = np.arange(t_total * i + 7 * t_swp_1 + t_swp_2, t_total * i + 8 * t_swp_1 + t_swp_2 + t_swp_1/3, t_swp_1/3)
            knots_10 = np.arange(t_total * i + 8 * t_swp_1 + t_swp_2, t_total * i + 9 * t_swp_1 + t_swp_2 + t_swp_1/3, t_swp_1/3)
            knots_11 = np.arange(t_total * i + 9 * t_swp_1 + t_swp_2, t_total * i + 10 * t_swp_1 + t_swp_2 + t_swp_1/3, t_swp_1/3)
            knots_12 = np.arange(t_total * i + 10 * t_swp_1 + t_swp_2, t_total * i + 11 * t_swp_1 + t_swp_2 + t_swp_1/3, t_swp_1/3)
            knots_13 = np.arange(t_total * i + 11 * t_swp_1 + t_swp_2, t_total * i + 12 * t_swp_1 + t_swp_2 + t_swp_1/3, t_swp_1/3)
            knots_14 = np.arange(t_total * i + 12 * t_swp_1 + t_swp_2, t_total * i + 13 * t_swp_1 + t_swp_2 + t_swp_1/3, t_swp_1/3)
        else:
            ind = int(i % 5)
            t_swp = lens[ind]
            df_rep = df_all[(df_all['t'] >= t_total) & (df_all['t'] < t_total + t_swp)][['t', 'x']]
            knots = np.arange(t_total, t_total + t_swp, t_swp/2)
            t_total += t_swp
        if data == '24102024_MA_FP_RT':
            bsplines_1 = skfda.representation.basis.BSplineBasis(knots=knots_1, order=n_order)
            bsplines_2 = skfda.representation.basis.BSplineBasis(knots=knots_2, order=n_order)
            phi_1 = np.array(bsplines_1(df_rep_1['t'].values)).T[0]
            phi_2 = np.array(bsplines_2(df_rep_2['t'].values)).T[0]
            operator = skfda.misc.operators.LinearDifferentialOperator(2)
            regularization = skfda.misc.regularization.L2Regularization(operator)
            r_1 = regularization.penalty_matrix(bsplines_1)
            r_2 = regularization.penalty_matrix(bsplines_2)
            m_1 = solve(phi_1.T @ phi_1 + lambda_ * r_1, phi_1.T)
            m_2 = solve(phi_2.T @ phi_2 + lambda_ * r_2, phi_2.T)
            c_hat_1 = m_1 @ df_rep_1['x'].values
            c_hat_2 = m_2 @ df_rep_2['x'].values
            x_hat_1 = phi_1 @ c_hat_1
            x_hat_2 = phi_2 @ c_hat_2
            all_.extend(x_hat_1)
            all_.extend(x_hat_2)
        elif data == '20241114_MA_FP_RT':
            for knots, df_rep in zip([knots_1,knots_2,knots_3,knots_4,knots_5,knots_6],
                                     [df_rep_1,df_rep_2,df_rep_3,df_rep_4,df_rep_5,df_rep_6]):
                bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=n_order)
                phi = np.array(bsplines(df_rep['t'].values)).T[0]
                operator = skfda.misc.operators.LinearDifferentialOperator(2)
                regularization = skfda.misc.regularization.L2Regularization(operator)
                r = regularization.penalty_matrix(bsplines)
                m = solve(phi.T @ phi + lambda_ * r, phi.T)
                c_hat = m @ df_rep['x'].values
                x_hat = phi @ c_hat
                all_.extend(x_hat)
        elif data == '20241128_MA_FP_RT':
            for knots, df_rep in zip([knots_1,knots_2,knots_3],
                                     [df_rep_1,df_rep_2,df_rep_3]):
                bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=n_order)
                phi = np.array(bsplines(df_rep['t'].values)).T[0]
                operator = skfda.misc.operators.LinearDifferentialOperator(2)
                regularization = skfda.misc.regularization.L2Regularization(operator)
                r = regularization.penalty_matrix(bsplines)
                m = solve(phi.T @ phi + lambda_ * r, phi.T)
                c_hat = m @ df_rep['x'].values
                x_hat = phi @ c_hat
                all_.extend(x_hat)
        elif data == '20241128_MA_FP_RT_2':
            for knots, df_rep in zip([knots_1,knots_2,knots_3,knots_4,knots_5,knots_6,knots_7,knots_8,knots_9,knots_10,knots_11,knots_12,knots_13,knots_14],
                                     [df_rep_1,df_rep_2,df_rep_3,df_rep_4,df_rep_5,df_rep_6,df_rep_7,df_rep_8,df_rep_9,df_rep_10,df_rep_11,df_rep_12,df_rep_13,df_rep_14]):
                bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=n_order)
                phi = np.array(bsplines(df_rep['t'].values)).T[0]
                operator = skfda.misc.operators.LinearDifferentialOperator(2)
                regularization = skfda.misc.regularization.L2Regularization(operator)
                r = regularization.penalty_matrix(bsplines)
                m = solve(phi.T @ phi + lambda_ * r, phi.T)
                c_hat = m @ df_rep['x'].values
                x_hat = phi @ c_hat
                all_.extend(x_hat)
        else:
            bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=n_order)
            phi = np.array(bsplines(df_rep['t'].values)).T[0]

            # calculate penalty matrix
            operator = skfda.misc.operators.LinearDifferentialOperator(2)
            regularization = skfda.misc.regularization.L2Regularization(operator)
            r = regularization.penalty_matrix(bsplines)

            m = solve(phi.T @ phi + lambda_ * r, phi.T)
            c_hat = m @ df_rep['x'].values
            x_hat = phi @ c_hat

            all_.extend(x_hat)

    # Save output
    while len(all_) < len(df_all['t']):
        last = all_[-1]
        all_.append(last)
    pd.DataFrame(all_).to_csv(f"{output}/synth_Y_fit.csv", index=False)

if __name__ == "__main__":
    t_steps = ast.literal_eval(args.s)
    main(args.i, args.o, float(args.l), t_steps, args.m, args.t)
