## Imports
import os, sys
import time
import yaml
import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

exp_id = 'fdp1'  #<--- Select 1 out of 4 Physiologies (fdp1-fdp4)
parameter_set_dim = 411  # No of kinetic parameters in model

path_parameters = f'models/parameters/parameters_sample_id_{exp_id}_0.hdf5'
if not path_parameters.endswith('.hdf5'):
    raise ValueError('Your data must be a .hdf5 file')

# fetch training set eigenvalues
path_stability = f'models/parameters/maximal_eigenvalues_{exp_id}.csv'
if not path_stability.endswith('.csv'):
    raise ValueError('Your data must be a .csv file')

def main():
    # get the data and processed
    f = h5py.File(path_parameters, 'r')
    stabilities = pd.read_csv(path_stability).iloc[:, 1].values

    n_parameters = f[('num_parameters_sets')][()]
    all_data = np.empty([n_parameters, parameter_set_dim])
    all_stabilities = np.empty([n_parameters])

    J_partition = -9  #<--- Create class partition based on this eigenvalue
    count0, count1 = 0, 0

    for i in range(0, n_parameters):

        if i % 10000 == 0:
            print(f'current set processed: {i}')
        this_param_set = f'parameter_set_{i}'
        param_values = np.array(f.get(this_param_set))

        mreal = stabilities[i]

        if mreal >= J_partition:
            stability = 1
            count0 += 1
        elif mreal < J_partition:
            stability = -1
            count1 += 1

        all_data[i] = param_values
        all_stabilities[i] = stability

    all_data = np.array(all_data)
    all_stabilities = np.array(all_stabilities)

    n_param = all_data.shape[0]
    print(f'% relevant models: {count1 / n_param}')

    # keep only km
    parameter_names = list(f['parameter_names'])
    # bytes to str
    parameter_names = [x.decode('utf-8') for x in parameter_names]
    idx_to_keep = [i for i, x in enumerate(parameter_names) if 'km_' in x]
    all_km = all_data[:, idx_to_keep]
    all_km_names = [x for i, x in enumerate(parameter_names) if 'km_' in x]

    print(f'Shape of all data: {all_km.shape}')

    all_stabilities = pd.DataFrame(all_stabilities, columns=['Stability'])
    all_stabilities.to_csv('../data/rekindle_stabilities.csv', index=False)

    data = pd.DataFrame(all_data, columns=parameter_names)
    data.to_csv('../data/rekindle_data.csv', index=False)
    
if __name__ == '__main__':
    main()
    print('Preprocessing Complete')