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

parser = argparse.ArgumentParser(description='Spline fitting')
parser.add_argument('-i', type=str, required=True, help='Input folder')
parser.add_argument('-o', type=str, required=True, help='Output folder')
parser.add_argument('-l', type=float, default=5, help='Smoothing parameter lambda')
parser.add_argument('-m', type=str, default='milnes', help='Milnes or optimal')
args = parser.parse_args()

def main(input, output, lambda_, data):
    df_all = pd.read_csv(f"{input}/synth_Y.csv")

    # Parameters
    if data == 'milnes':
        t_swp = 10000
        swps = sweeps
    else:
        t_swp = 3340
        t_total = 0
        swps = sweeps*3
    n_order = 4
    all_ = []

    # Fit splines
    for i in range(swps):
        if data == 'milnes':
            df_rep = df_all[(df_all['t'] >= t_swp * i) & (df_all['t'] < t_swp * (i + 1))][['t', 'x']]
            knots = np.arange(t_swp * i, t_swp * (i + 1) + t_swp/2, t_swp/2)
        else:
            df_rep = df_all[(df_all['t'] >= t_total) & (df_all['t'] < t_total + t_swp)][['t', 'x']]
            knots = np.arange(t_total, t_total + t_swp, t_swp/2)
            t_total += t_swp
            if t_swp == 3340:
                t_swp = 3330
            elif (t_total % 10000 == 0):
                t_swp = 3340
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
    pd.DataFrame(all_).to_csv(f"{output}/synth_Y_fit.csv", index=False)

if __name__ == "__main__":
    main(args.i, args.o, float(args.l), args.m)
