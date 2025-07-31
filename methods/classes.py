import pints
import numpy as np
from scipy import special
from .models import Model
from . import sweeps, T_default
from matplotlib import pyplot as plt

### Define normal ratio likelihood
class NormalRatioLogLikelihood(pints.ProblemLogLikelihood):
    '''
    Calculates the normal ratio log-likelihood for a given parameter set x and some 
    data assumed to be the ratio of two normal random variables.
    Requires some approximation of the control (i.e. spline fit) mu_y.
    '''
    def __init__(self, problem, mu_y, td = False, multi = False, all = False, conc = 0):
        super(NormalRatioLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()
        self._conc = conc

        # Get control spline fit
        self._mu_y = mu_y

        self._td = td

        # Add parameters to problem
        if self._td:
            if multi:
                self._n_parameters = problem.n_parameters() + 6 + self._no
            elif all:
                self._n_parameters = problem.n_parameters() + 9 + self._no
            else:
                self._n_parameters = problem.n_parameters() + 3 + self._no
        else:
            self._n_parameters = problem.n_parameters() + self._no

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        beta = self._problem.evaluate(x[:-self._no]) # evaluate the model
        if any(sigma <= 0): #require positive standard deviation (sd)
            return -np.inf
        # assume sd is the same for both control and drug sweeps
        sigma_x = sigma
        sigma_y = sigma
        rho = sigma_x/sigma_y
        delta = sigma_y/self._mu_y  #mu_y spline fit to control
        z = self._values # get the data
        #z = np.ones(len(self._values))

        # calculate pdf and return the sum of log(pdf)
        q = (1+beta*rho**2*z)/(delta*np.sqrt(1+rho**2*z**2))

        pdf = rho/(np.pi*(1+(rho**2)*(z**2)))*(np.exp(-(rho**2*beta**2+1)/(2*(delta**2))) + np.sqrt(np.pi/2)*q*special.erf(q/np.sqrt(2))*np.exp(-rho**2*(z-beta)**2/(2*(delta**2)*(1+rho**2*z**2))))

        #sign_pdf = []
        #above = 0
        #below = 0
        #for p,z_i,b in zip(pdf,z,beta):
        #    if z_i-b<0:
        #        sign_pdf.append(p)
        #        above+=p
        #    elif z_i-b>0:
        #        sign_pdf.append(-p)
        #        below+=p
        #fig, (ax1, ax2) = plt.subplots(2,1,figsize = (14, 7))
        #ax1.set_title(f'{self._conc}nM')
        #ax1.plot(z, linewidth = 0.05, alpha = 0.8, label = 'data')
        #ax1.plot(beta, label = 'model fit')
        #ax1.set_ylim(-0.5, 1.5)
        #ax1.legend()
        #ax2.plot(sign_pdf, linewidth = 0.01, alpha = 0.5)
        #ax2.set_ylim(-14, 14)
        #ax2.set_title(f'{np.round(above,2)} above, {np.round(below,2)} below')
        #plt.savefig(f'pdf_plot_{self._conc}nM.png')

        # identify indices of 'inf' values in pdf
        #inf_ind = np.where(np.isinf(np.log(pdf)))[0]

        # sum finite values
        #finite_vals = np.log(pdf)[~np.isinf(np.log(pdf))]
        #finite_sum = np.sum(finite_vals)

        # evaluate approximation for 'inf' values and add to sum
        #for i in inf_ind:
        #    finite_sum += np.log(1/np.pi) + np.log(rho/(1+rho**2*z[i]**2)) - (rho**2*(z[i]-beta[i])**2)/(2*(delta[i]**2)*(1+rho**2*z[i]**2)) + np.sqrt(np.pi/2)*q[i]*special.erf(q[i]/np.sqrt(2))

        #if len(inf_ind) > 0:
        #    finite_sum = finite_sum[0]

        return np.sum(np.log(pdf))

### Define PINTS Model
class ConcatMilnesModel(pints.ForwardModel):
    """A PINTS model simulating concatenated protocol."""
    def __init__(self, model, protocol, times, win, conc, param_dict, additional_pars=0, multi=False, all=False):
        self._model = Model(model,
                            protocol,
                            parameters=['binding'],
                            analytical=True)
        # TODO currently hardcoded to get model number
        if model.split("-")[0] != 'kemp' and model.split("-")[0] != 'sis':
            if len(model.split("-"))<=1:
                model_num = model[1:]
        elif len(model.split("-"))>1 and model.split("-")[0] != 'kemp' and model.split("-")[0] != 'sis':
            model_num = model.split("-")[0][1:]
        else:
            model_num = model.split("-")[1][1:]
        if model_num in ['12', '13']:
            self._model.fix_kt()
        self._win = win
        self._conc = conc
        self._model_name = model
        self._additional_pars = additional_pars
        self._multi = multi
        self._all = all
        # TODO currently hardcoded to get number of pulses
        if (model.split("-")[0] == 'sis' or len(model.split("-"))>1) and model.split("-")[0] != 'kemp':
            if protocol == "protocols/3_drug_protocol_23_10_24.mmt":
                self.n_pulses = 13
            elif protocol == "protocols/3_drug_protocol_14_11_24.mmt":
                self.n_pulses = 10
            elif protocol == "protocols/Milnes_16102024_MA1_FP_RT.mmt":
                self.n_pulses = 10
            elif protocol == "protocols/gary_manual.mmt":
                self.n_pulses = 9
            else:
                self.n_pulses = 10
        #elif times[-1] != 14999.5:
        #    self.n_pulses = int(np.floor(250000/times[-1]))
        else:
            self.n_pulses = sweeps
        self._times = times
        # Simulate dose free (control)
        self._model.set_dose(0)
        self._model.set_temperature(T_default)
        # TODO currently hardcoded whether we need to set parameters 
        # or if they are already specified correctly in the model file
        if model.split("-")[0] != 'kemp' and model.split("-")[0] != 'sis':
            if len(model.split("-"))<=1:
                self._model.set_fix_parameters(param_dict)
        # Initialise control
        self._z = np.ones(self._model.n_parameters())
        self._before = self._model.simulate(self._z, self._times)[self._win]

    def n_parameters(self):
        return self._model.n_parameters()

    def simulate(self, parameters, times):
        if len(self._model_name.split("-")) > 1:
            if self._model_name.split("-")[1] == 'td': # or self._model_name.split("-")[0] == 'sis':
                self._model.set_temperature(parameters[-3-self._additional_pars])
                self._model.set_dose(0)
                self._before = self._model.simulate(self._z, self._times)[self._win]
        # Set concentration
        self._model.set_dose(self._conc)
        try:
            # Loop and append proportion open model output for each pulse
            out = []
            if len(self._model_name.split("-")) > 1:
                if self._model_name.split("-")[1] == 'td': # or self._model_name.split("-")[0] == 'sis':
                    self._model.set_temperature(parameters[-2-self._additional_pars])
                    if self._multi:
                        after = self._model.simulate(parameters[:-6-self._additional_pars], self._times)
                    elif self._all:
                        after = self._model.simulate(parameters[:-9], self._times)
                    else:
                        after = self._model.simulate(parameters[:-3-self._additional_pars], self._times)
            else:
                after = self._model.simulate(parameters, self._times)
            out = np.append(out, after[self._win] / self._before)
            for i in range(self.n_pulses - 1):
                if len(self._model_name.split("-")) > 1:
                    if self._model_name.split("-")[1] == 'td': # or self._model_name.split("-")[0] == 'sis':
                        self._model.set_temperature(parameters[-2-self._additional_pars]+np.log(i+2)*parameters[-1-self._additional_pars])
                        if self._multi:
                            after = self._model.simulate(parameters[:-6-self._additional_pars], self._times, reset=False)
                        elif self._all:
                            after = self._model.simulate(parameters[:-9], self._times, reset=False)
                        else:
                            after = self._model.simulate(parameters[:-3-self._additional_pars], self._times, reset=False)
                else:
                    after = self._model.simulate(parameters, self._times, reset=False)
                out = np.append(out, after[self._win] / self._before)
        except:
            out = np.ones(times.shape) * float('inf')
        assert(len(out) == len(times))
        return out
