import numpy as np
import matplotlib.pyplot as plt
import sys, os
import glob

d_path = '/Users/angeludr/Documents/GitHub/sma_phasing_sims/vlbi_cal/v1'
npy_file = '*.npy'

swarm_2017 = glob.glob(d_path + '/2017/' + npy_file, recursive=False)
swarm_2018 = glob.glob(d_path + '/2018/' + npy_file, recursive=False) 

print(swarm_2018)