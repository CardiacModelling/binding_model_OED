import pints
import numpy as np
from scipy import special
from .models import Model
from . import sweeps

### Define normal ratio likelihood
class NormalRatioLogLikelihood(pints.ProblemLogLikelihood):

    def __init__(self, problem, mu_y):
        super(NormalRatioLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Get control spline fit
        self._mu_y = mu_y

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        if any(sigma <= 0):
            return -np.inf
        sigma_x = sigma
        sigma_y = sigma
        rho = sigma_x/sigma_y
        beta = self._problem.evaluate(x[:-self._no])
        delta = sigma_y/self._mu_y  #mu_y spline fit to control
        z = self._values
        q = (1+beta*rho**2*z)/(delta*np.sqrt(1+rho**2*z**2))
        pdf = rho/(np.pi*(1+z**2))*(np.exp(-(rho**2*beta**2+1)/(2*(delta**2))) + np.sqrt(np.pi/2)*q*special.erf(q/np.sqrt(2))*np.exp(-rho**2*(z-beta)**2/(2*(delta**2)*(1+rho**2*z**2))))
        return np.sum(np.log(pdf))

### Define PINTS Model
class ConcatMilnesModel(pints.ForwardModel):
    """A PINTS model simulating concatenated Milnes protocol."""
    def __init__(self, model, protocol, times, win, conc, param_dict):
        self._model =  Model(model,
                            protocol,
                            parameters=['binding'],
                            analytical=True)
        model_num = model[1:]
        if model_num in ['12', '13']:
            self._model.fix_kt()
        self._win = win
        self._conc = conc
        if times[-1] > 15e3:
            self.n_pulses = 5
        else:
            self.n_pulses = sweeps
        self._times = times
        # Simulate dose free (control)
        self._model.set_dose(0)
        self._model.set_fix_parameters(param_dict)
        z = np.ones(self._model.n_parameters())
        self._before = self._model.simulate(z, self._times)[self._win]

    def n_parameters(self):
        return self._model.n_parameters()

    def simulate(self, parameters, times):
        self._model.set_dose(self._conc)
        try:
            out = []
            after = self._model.simulate(parameters, self._times)
            out = np.append(out, after[self._win] / self._before)
            for i in range(self.n_pulses - 1):
                after = self._model.simulate(parameters, self._times, reset=False)
                out = np.append(out, after[self._win] / self._before)
        except:
            out = np.ones(times.shape) * float('inf')
        assert(len(out) == len(times))
        return out
