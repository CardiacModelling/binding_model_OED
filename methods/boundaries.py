#
# PINTS Boundaries that limit the transition rates.
#
import numpy as np
import pints


class Boundaries(pints.Boundaries):
    """
    Boundary constraints on the parameters
    """
    def __init__(self, which_model='1', fix_hill=False, likelihood=False, ts=1, td=False, multi=False, all=False):
        '''
        which_model: (str) model number (e.g. '1')
        fix_hill: (bool) exclude the Hill coefficient if True.
        '''
        super(Boundaries, self).__init__()

        self.which_model = which_model
        self.fix_hill = fix_hill
        self.likelihood = likelihood
        self.ts = ts
        self.td = td
        self.multi = multi
        self.all = all

        self._lower_kon = 1e-9
        self._upper_kon = 1

        self._lower_ku = 1e-9
        self._upper_ku = 1e3

        self._lower_kt = 1e-9
        self._upper_kt = 1e3

        self._lower_k2f = 1e-9
        self._upper_k2f = 1

        self._lower_hill = 0.2
        self._upper_hill = 2

        self._lower_kmax_fda = 1e-1
        self._upper_kmax_fda = 1e8

        self._lower_ku_fda = 1e-5
        self._upper_ku_fda = 1e1

        self._lower_halfmax_fda = 1e0
        self._upper_halfmax_fda = 1e9

        self._lower_vhalf_fda = 0
        self._upper_vhalf_fda = 2e2

        self._lower_sigma = 1e-2
        self._upper_sigma = 1e2

        self._lower_controlT = 299.15
        self._upper_controlT = 301.15

        self._lower_initdrugT = 295.15
        self._upper_initdrugT = 301.15

        self._lower_Trange = 1e-9
        self._upper_Trange = 10

        if self.multi or self.all:
            self._lower_controlT_m = 303.15
            self._upper_controlT_m = 305.15

            self._lower_initdrugT_m = 295.15
            self._upper_initdrugT_m = 305.15
            if self.all:
                self._lower_controlT_o = 293.15
                self._upper_controlT_o = 295.15

                self._lower_initdrugT_o = 292.15
                self._upper_initdrugT_o = 295.15

        self._lower = np.array([
            self._lower_kon,
            self._lower_kon
        ])

        self._upper = np.array([
            self._upper_kon,
            self._upper_kon
        ])

        if self.which_model in {'0a', '0b', '1', '2', '2i', '3', '4', '5', '5i'}:
            if self.fix_hill:
                n_params = 2
            else:
                self._lower = np.append(self._lower, self._lower_hill)
                self._upper = np.append(self._upper, self._upper_hill)
                n_params = 3
        elif self.which_model in {'8', '9'}:
            if self.fix_hill:
                self._lower = np.append(self._lower, self._lower_kon)
                self._upper = np.append(self._upper, self._upper_kon)
                n_params = 3
            else:
                self._lower = np.append(self._lower, [self._lower_kon, self._lower_hill])
                self._upper = np.append(self._upper, [self._upper_kon, self._upper_hill])
                n_params = 4
        elif self.which_model == '6':
            if self.fix_hill:
                self._lower = np.append(self._lower, self._lower_k2f)
                self._upper = np.append(self._upper, self._upper_k2f)
                n_params = 3
            else:
                self._lower = np.append(self._lower, [self._lower_k2f, self._lower_hill])
                self._upper = np.append(self._upper, [self._upper_k2f, self._upper_hill])
                n_params = 4
        elif self.which_model == '10':
            if self.fix_hill:
                self._lower = np.append(self._lower, [self._lower_kon, self._lower_k2f])
                self._upper = np.append(self._upper, [self._upper_kon, self._upper_k2f])
                n_params = 4
            else:
                self._lower = np.append(self._lower, [self._lower_kon, self._lower_k2f, self._lower_hill])
                self._upper = np.append(self._upper, [self._upper_kon, self._upper_k2f, self._upper_hill])
                n_params = 5
        elif self.which_model == '11':
            if self.fix_hill:
                self._lower = np.append(self._lower, [self._lower_ku, self._lower_kt])
                self._upper = np.append(self._upper, [self._upper_ku, self._upper_kt])
                n_params = 4
            else:
                self._lower = np.append(self._lower, [self._lower_ku, self._lower_kt, self._lower_hill])
                self._upper = np.append(self._upper, [self._upper_ku, self._upper_kt, self._upper_hill])
                n_params = 5
        elif self.which_model == '7':
            if self.fix_hill:
                self._lower = np.append(self._lower, [self._lower_kon, self._lower_kon])
                self._upper = np.append(self._upper, [self._upper_kon, self._upper_kon])
                n_params = 4
            else:
                self._lower = np.append(self._lower, [self._lower_kon, self._lower_kon, self._lower_hill])
                self._upper = np.append(self._upper, [self._upper_kon, self._upper_kon, self._upper_hill])
                n_params = 5
        elif self.which_model == '12':
            # TODO?
            if self.fix_hill:
                self._lower = np.array([self._lower_kmax_fda, self._lower_ku_fda, self._lower_halfmax_fda, self._lower_vhalf_fda])
                self._upper = np.array([self._upper_kmax_fda, self._upper_ku_fda, self._upper_halfmax_fda, self._upper_vhalf_fda])
                n_params = 4
            else:
                self._lower = np.array([self._lower_kmax_fda, self._lower_ku_fda, self._lower_halfmax_fda, self._lower_hill, self._lower_vhalf_fda])
                self._upper = np.array([self._upper_kmax_fda, self._upper_ku_fda, self._upper_halfmax_fda, self._upper_hill, self._upper_vhalf_fda])
                n_params = 5
        elif self.which_model == '13':
            # TODO?
            if self.fix_hill:
                self._lower = np.array([self._lower_kon, self._lower_ku_fda, self._lower_vhalf_fda])
                self._upper = np.array([self._upper_kon, self._upper_ku_fda, self._upper_vhalf_fda])
                n_params = 3
            else:
                self._lower = np.array([self._lower_kon, self._lower_ku_fda, self._lower_hill, self._lower_vhalf_fda])
                self._upper = np.array([self._upper_kon, self._upper_ku_fda, self._upper_hill, self._upper_vhalf_fda])
                n_params = 4
        else:
            raise ValueError(f'Unknown model {self.which_model}')

        if self.td:
            self._lower = np.append(self._lower, [self._lower_controlT, self._lower_initdrugT, self._lower_Trange])
            self._upper = np.append(self._upper, [self._upper_controlT, self._upper_initdrugT, self._upper_Trange])
            n_params+=3
            if self.multi or self.all:
                self._lower = np.append(self._lower, [self._lower_controlT_m, self._lower_initdrugT_m, self._lower_Trange])
                self._upper = np.append(self._upper, [self._upper_controlT_m, self._upper_initdrugT_m, self._upper_Trange])
                n_params+=3
                if self.all:
                    self._lower = np.append(self._lower, [self._lower_controlT_o, self._lower_initdrugT_o, self._lower_Trange])
                    self._upper = np.append(self._upper, [self._upper_controlT_o, self._upper_initdrugT_o, self._upper_Trange])
                    n_params+=3

        if self.likelihood:
            for i in range(0, self.ts):
                self._lower = np.append(self._lower, self._lower_sigma)
                self._upper = np.append(self._upper, self._upper_sigma)
                n_params+=1

        self._upper = pints.vector(self._upper)
        self._lower = pints.vector(self._lower)

        self._n_params = n_params

    def n_parameters(self):
        return self._n_params

    def check(self, parameters):

        debug = False

        # Check parameter boundaries
        if np.any(parameters < self._lower):
            if debug: print('Lower')
            return False
        if np.any(parameters > self._upper):
            if debug: print('Upper')
            return False

        return True

    def sample(self, n=1, sample_with_log_transform=True):

        if n > 1:
            raise NotImplementedError

        p = np.zeros(self._n_params)

        if sample_with_log_transform:
            f = lambda x: np.log(x)
            unf = lambda x: np.exp(x)
        else:
            f = lambda x: x
            unf = lambda x: x

        # Sample parameters
        p[0] = unf(np.random.uniform(
            f(self._lower_kon), f(self._upper_kon)))
        p[1] = unf(np.random.uniform(
            f(self._lower_kon), f(self._upper_kon)))

        if self.which_model in {'0a', '0b', '1', '2', '2i', '3', '4', '5', '5i'}:
            if self.fix_hill:
                if self.likelihood:
                    if self.td:
                        p[2] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[3] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[4] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[5] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[6] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[7] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[8] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[9] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[10] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+11] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+8] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+5] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+2] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
                else:
                    pass
            else:
                p[2] = unf(np.random.uniform(
                    f(self._lower_hill), f(self._upper_hill)))
                if self.likelihood:
                    if self.td:
                        p[3] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[4] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[5] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[6] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[7] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[8] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[9] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[10] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[11] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+12] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+9] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+6] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+3] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
        elif self.which_model in {'8', '9'}:
            p[2] = unf(np.random.uniform(
                f(self._lower_kon), f(self._upper_kon)))
            if self.fix_hill:
                if self.likelihood:
                    if self.td:
                        p[3] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[4] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[5] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[6] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[7] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[8] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[9] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[10] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[11] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+12] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+9] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+6] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+3] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
                else:
                    pass
            else:
                p[3] = unf(np.random.uniform(
                    f(self._lower_hill), f(self._upper_hill)))
                if self.likelihood:
                    if self.td:
                        p[4] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[5] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[6] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[7] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[8] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[9] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[10] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[11] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[12] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+13] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+10] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+7] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+4] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))

        elif self.which_model == '6':
            p[2] = unf(np.random.uniform(
                f(self._lower_k2f), f(self._upper_k2f)))
            if self.fix_hill:
                if self.likelihood:
                    if self.td:
                        p[3] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[4] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[5] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[6] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[7] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[8] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[9] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[10] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[11] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+12] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+9] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+6] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+3] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
                else:
                    pass
            else:
                p[3] = unf(np.random.uniform(
                    f(self._lower_hill), f(self._upper_hill)))
                if self.likelihood:
                    if self.td:
                        p[4] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[5] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[6] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[7] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[8] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[9] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[10] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[11] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[12] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+13] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+10] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+7] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+4] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
        elif self.which_model == '10':
            p[2] = unf(np.random.uniform(
                f(self._lower_kon), f(self._upper_kon)))
            p[3] = unf(np.random.uniform(
                f(self._lower_k2f), f(self._upper_k2f)))
            if self.fix_hill:
                if self.likelihood:
                    if self.td:
                        p[4] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[5] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[6] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[7] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[8] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[9] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[10] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[11] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[12] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+13] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+10] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+7] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+4] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
                else:
                    pass
            else:
                p[4] = unf(np.random.uniform(
                    f(self._lower_hill), f(self._upper_hill)))
                if self.likelihood:
                    if self.td:
                        p[5] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[6] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[7] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[8] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[9] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[10] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[11] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[12] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[13] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+14] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+11] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+8] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+5] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
        elif self.which_model == '11':
            p[2] = unf(np.random.uniform(
                f(self._lower_ku), f(self._upper_ku)))
            p[3] = unf(np.random.uniform(
                f(self._lower_kt), f(self._upper_kt)))
            if self.fix_hill:
                if self.likelihood:
                    if self.td:
                        p[4] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[5] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[6] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[7] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[8] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[9] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[10] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[11] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[12] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+13] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+10] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+7] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+4] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
                else:
                    pass
            else:
                p[4] = unf(np.random.uniform(
                    f(self._lower_hill), f(self._upper_hill)))
                if self.likelihood:
                    if self.td:
                        p[5] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[6] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[7] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[8] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[9] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[10] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[11] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[12] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[13] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+14] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+11] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+8] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+5] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
        elif self.which_model == '7':
            p[2] = unf(np.random.uniform(
                f(self._lower_kon), f(self._upper_kon)))
            p[3] = unf(np.random.uniform(
                f(self._lower_kon), f(self._upper_kon)))
            if self.fix_hill:
                if self.likelihood:
                    if self.td:
                        p[4] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[5] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[6] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[7] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[8] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[9] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[10] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[11] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[12] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+13] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+10] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+7] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+4] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
                else:
                    pass
            else:
                p[4] = unf(np.random.uniform(
                    f(self._lower_hill), f(self._upper_hill)))
                if self.likelihood:
                    if self.td:
                        p[5] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[6] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[7] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[8] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[9] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[10] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[11] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[12] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[13] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+14] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+11] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+8] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+5] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
        elif self.which_model == '12':
            p[0] = unf(np.random.uniform(f(self._lower_kmax_fda), f(self._upper_kmax_fda)))
            p[1] = unf(np.random.uniform(f(self._lower_ku_fda), f(self._upper_ku_fda)))
            p[2] = unf(np.random.uniform(f(self._lower_halfmax_fda), f(self._upper_halfmax_fda)))
            if self.fix_hill:
                p[3] = np.random.uniform(self._lower_vhalf_fda, self._upper_vhalf_fda)
                if self.likelihood:
                    if self.td:
                        p[4] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[5] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[6] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[7] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[8] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[9] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[10] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[11] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[12] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+13] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+10] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+7] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+4] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
            else:
                p[3] = unf(np.random.uniform(f(self._lower_hill), f(self._upper_hill)))
                p[4] = np.random.uniform(self._lower_vhalf_fda, self._upper_vhalf_fda)
                if self.likelihood:
                    if self.td:
                        p[5] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[6] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[7] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[8] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[9] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[10] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[11] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[12] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[13] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+14] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+11] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+8] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+5] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))

        elif self.which_model == '13':
            p[0] = unf(np.random.uniform(f(self._lower_kon), f(self._upper_kon)))
            p[1] = unf(np.random.uniform(f(self._lower_ku_fda), f(self._upper_ku_fda)))
            if self.fix_hill:
                p[2] = np.random.uniform(self._lower_vhalf_fda, self._upper_vhalf_fda)
                if self.likelihood:
                    if self.td:
                        p[3] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[4] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[5] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[6] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[7] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[8] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[9] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[10] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[11] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+12] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+9] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+6] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+3] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
            else:
                p[2] = unf(np.random.uniform(f(self._lower_hill), f(self._upper_hill)))
                p[3] = np.random.uniform(self._lower_vhalf_fda, self._upper_vhalf_fda)
                if self.likelihood:
                    if self.td:
                        p[4] = np.random.uniform(self._lower_controlT, self._upper_controlT)
                        p[5] = np.random.uniform(self._lower_initdrugT, self._upper_initdrugT)
                        p[6] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                        if self.multi or self.all:
                            p[7] = np.random.uniform(self._lower_controlT_m, self._upper_controlT_m)
                            p[8] = np.random.uniform(self._lower_initdrugT_m, self._upper_initdrugT_m)
                            p[9] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                            if self.all:
                                p[10] = np.random.uniform(self._lower_controlT_o, self._upper_controlT_o)
                                p[11] = np.random.uniform(self._lower_initdrugT_o, self._upper_initdrugT_o)
                                p[12] = unf(np.random.uniform(f(self._lower_Trange), f(self._upper_Trange)))
                                for i in range(0, self.ts):
                                    p[i+13] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                            else:
                                for i in range(0, self.ts):
                                    p[i+10] = unf(np.random.uniform(
                                        f(self._lower_sigma), f(self._upper_sigma)))
                        else:
                            for i in range(0, self.ts):
                                p[i+7] = unf(np.random.uniform(
                                    f(self._lower_sigma), f(self._upper_sigma)))
                    else:
                        for i in range(0, self.ts):
                            p[i+4] = unf(np.random.uniform(
                                f(self._lower_sigma), f(self._upper_sigma)))
        else:
            pass

        # The Boundaries interface requires a matrix ``(n, n_parameters)``
        p.reshape(1, self._n_params)

        return p

    def lower(self):
        """
        Returns the lower boundaries for all parameters (as a read-only NumPy
        array).
        """
        return self._lower

    def range(self):
        """
        Returns the size of the parameter space (i.e. ``upper - lower``).
        """
        return self._upper - self._lower

    def upper(self):
        """
        Returns the upper boundary for all parameters (as a read-only NumPy
        array).
        """
        return self._upper
