#
# Methods module
#

# Get methods directory
import os, inspect
frame = inspect.currentframe()
DIR_METHOD = os.path.abspath(os.path.dirname(inspect.getfile(frame)))
del(os, inspect, frame)

concentrations = {
    'K_i': 140,
    'K_o': 5,
}

results = 'results'

t_hold = 10e3  # ms
v_hold = -80   # mV

herg_pars = ['ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4', 'ikr.p5',
             'ikr.p6', 'ikr.p7', 'ikr.p8', 'ikr.p9']

concs = [1, 3, 10, 30]
colrs = [f'C{i}' for i in range(len(concs))]
sweeps = 10
steps = 0.5
sd = 10

all_model_nums = ['1', '2', '2i', '3', '4', '5', '5i', '6',
                  '7', '8', '9', '10', '11', '12', '13']

