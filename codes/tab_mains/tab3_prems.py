"""Table of emissions premia for different policy responses and delayed
sectors.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.16.2024

To run: python tab3_params.py [cal] [save_output]
"""

import os
import sys

import numpy as np
import xarray as xr
import pandas as pd

# parse command line input
cal = sys.argv[1]
save_output = int(sys.argv[2])

# get data and open dataset
path = os.getcwd()
datapath = path + "/data/output/"

file_2 = cal + '_base2_analytics.nc'
file_3 = cal + '_base3_analytics.nc'
file_4 = cal + '_base4_analytics.nc'

ds2 = xr.open_dataset(datapath + file_2, engine='netcdf4')
ds3 = xr.open_dataset(datapath + file_3, engine='netcdf4')
ds4 = xr.open_dataset(datapath + file_4, engine='netcdf4')

energy_ind = 5
industry_ind = 1

prems_en_2 = [ds2.Bp.values[energy_ind, -1],
      ds2.Bp.values[energy_ind, ds2.delay_amount.values==5.0][0]]

prems_ind_2 = [ds2.Bp.values[industry_ind, -1],
      ds2.Bp.values[industry_ind, ds2.delay_amount.values==5.0][0]]


prems_en_3 = [ds3.Bp.values[energy_ind, -1],
      ds3.Bp.values[energy_ind, ds3.delay_amount.values==5.0][0]]

prems_ind_3 = [ds3.Bp.values[industry_ind, -1],
      ds3.Bp.values[industry_ind, ds3.delay_amount.values==5.0][0]]

prems_4 = [ds4.Bp.values[ds4.delay_amount.values==5.0][0], ds4.Bp.values[-1]]

prems = np.zeros((3, 3, 2))

prems[0, 0] = prems_en_2
prems[0, 1] = prems_ind_2
prems[1, 0] = prems_en_3
prems[1, 1] = prems_ind_3
prems[2, 2] = prems_4

ds_prem = xr.Dataset(data_vars={
    'emis_prem': (['policy_response', 'chal_sec', 'delay'], prems)
},
    coords={
        'delay': (['delay'], [5, 10]),
        'chal_sec': (['chal_sec'], ['Energy', 'Industry', 'All']),
        'policy_response': (['policy_response'], ['2', '3', '4'])}
)

if save_output:
    cwd = os.getcwd()
    filename = cwd + '/data/output/' + cal + "_emis_prems.nc"
    ds_prem.to_netcdf(filename, engine='netcdf4')
    print("Emissions premiums data saved to:\n{}".format(filename))

else:
    print(ds_prem)
