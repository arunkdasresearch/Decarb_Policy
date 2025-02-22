"""Prints quoted numbers throughout paper.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.16.2024

To run: quoted_numbers.py [cal]
"""

import os
import sys

import numpy as np
import xarray as xr

# parse command line input
cal = sys.argv[1]

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

exp_string_1 = "Relative cost increase in Policy Response 2 for delaying energy by a decade: "
exp_string_2 = "Relative cost increase in Policy Response 2 for delaying industry by a decade: "
exp_string_3 = "Additional cost of Policy Response 2 for delaying energy by a decade: "
exp_string_4 = "Additional cost of Policy Response 2 for delaying industry by a decade: "
exp_string_5 = "Additional cost of Policy Response 4 for a decade of delay: "
exp_string_6 = "Relative cost of Policy Response 4 for a decade of delay: "

print(exp_string_1 + str(ds2.cost.values[energy_ind, -1]/ds2.optimal_cost))
print(exp_string_2 + str(ds2.cost.values[industry_ind, -1]/ds2.optimal_cost))

print(exp_string_3 + str(ds2.cost.values[energy_ind, -1] - ds2.optimal_cost) +
     " billion")
print(exp_string_4 + str(ds2.cost.values[industry_ind, -1] - ds2.optimal_cost)
     + " billion")

print(exp_string_5 + str(ds4.cost.values[-1] - ds4.optimal_cost) + ' billion')
print(exp_string_6 + str(ds4.cost.values[-1]/ds4.optimal_cost) + ' billion')

