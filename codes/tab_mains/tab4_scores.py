"""Analysis script of delay impact on cost.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.16.2024

To run: python tab4_scores.py [cal]
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.linear_model import LinearRegression


def get_score(param, data):
    """Get r^2 value for regression.

    Parameters
    ----------
    param: list
        params, X values for regression

    data: list
        the data, Y values for regression

    Returns
    -------
    shape: float
        r^2 value of regression
    """

    reg = LinearRegression().fit(np.array(param).reshape(-1, 1),
                                 data)
    shape = reg.score(np.array(param).reshape(-1, 1),
                      data)
    return shape


# parse command line input
cal = sys.argv[1]

# get data and open dataset
path = os.getcwd()

data_path = path + "/data/output/"
inputpath = path + "/data/cal/" + cal + "_secs.csv"

file_2 = cal + '_base2_analytics.nc'
file_3 = cal + '_base3_analytics.nc'
file_4 = cal + '_base4_analytics.nc'

ds2 = xr.open_dataset(data_path + file_2, engine='netcdf4')
ds3 = xr.open_dataset(data_path + file_3, engine='netcdf4')
ds4 = xr.open_dataset(data_path + file_4, engine='netcdf4')

param_df = pd.read_csv(inputpath)

# define delay indices
delay_ind_5 = 29
delay_ind_10 = -1

# parameter indices
del_ind = 2
abar_ind = 1
cbar_ind = 5

# get y values: difference in cost between delay by 5 or 10 years and optimal
d2_cost_delay5 = ds2.cost.values[:, delay_ind_5] / ds2.optimal_cost
d2_cost_delay10 = ds2.cost.values[:, delay_ind_10] / ds2.optimal_cost

d3_cost_delay5 = ds3.cost.values[:, delay_ind_5] / ds3.optimal_cost
d3_cost_delay10 = ds3.cost.values[:, delay_ind_10] / ds3.optimal_cost

# take log of cost params
cbars = np.array(param_df.values[cbar_ind][1:], dtype=float)
logcbars = np.log10(cbars)

# compute all scores
delta5_d2_score = get_score(param_df.values[del_ind][1:], d2_cost_delay5)
abar5_d2_score = get_score(param_df.values[abar_ind][1:], d2_cost_delay5)
cbar5_d2_score = get_score(logcbars,
                           d2_cost_delay5)

delta10_d2_score = get_score(param_df.values[del_ind][1:], d2_cost_delay10)
abar10_d2_score = get_score(param_df.values[abar_ind][1:], d2_cost_delay10)
cbar10_d2_score = get_score(logcbars,
                            d2_cost_delay10)

delta5_d3_score = get_score(param_df.values[del_ind][1:], d3_cost_delay5)
abar5_d3_score = get_score(param_df.values[abar_ind][1:], d3_cost_delay5)
cbar5_d3_score = get_score(logcbars,
                           d3_cost_delay5)

delta10_d3_score = get_score(param_df.values[del_ind][1:], d3_cost_delay10)
abar10_d3_score = get_score(param_df.values[abar_ind][1:], d3_cost_delay10)
cbar10_d3_score = get_score(logcbars,
                            d3_cost_delay10)

# make pretty table and print
table = [[2, 5, delta5_d2_score, abar5_d2_score, cbar5_d2_score],
         [2, 10, delta10_d2_score, abar10_d2_score, cbar10_d3_score],
         [3, 5, delta5_d3_score, abar5_d3_score, cbar5_d3_score],
         [3, 10, delta10_d3_score, abar10_d3_score, cbar10_d3_score]]

df = pd.DataFrame(data=table,
                  columns=['policy response', 'delay', 'delta', 'abar',
                           'cbar'])

print(df)
