"""Table 5: Warming crossings.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.28.2024

To run: python tab5_crossingtimes.py cal
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from src.expost_paths import get_abatement_path

# parse command line
cal = sys.argv[1]

# import data from simulations
cwd = os.getcwd()
data_path = cwd + '/data/output/'

file_2 = cal + '_base2_analytics.nc'
file_3 = cal + '_base3_analytics.nc'
file_4 = cal + '_base4_analytics.nc'

ds2 = xr.open_dataset(data_path + file_2, engine='netcdf4')
ds3 = xr.open_dataset(data_path + file_3, engine='netcdf4')
ds4 = xr.open_dataset(data_path + file_4, engine='netcdf4')

# import model parameters
path_glob = ''.join([cwd, '/data/cal/', cal, '_glob.csv'])
path_secs = ''.join([cwd, '/data/cal/', cal, '_secs.csv'])

df_glob = pd.read_csv(path_glob, delimiter=',', header=0)
df_secs = pd.read_csv(path_secs, delimiter=',', header=0, index_col=0)

# parse parameters for ease of use
r = df_glob['r'].values[0]

deltas = []
cbars = []
abars = []
for col in df_secs.columns:
    deltas.append(df_secs[col]['delta'])
    cbars.append(df_secs[col]['cbar'])
    abars.append(df_secs[col]['abar'])

# choose interesting cases
delay_sec1 = 5  # energy
delay_sec2 = 1  # industry
delay_amount_ind = -1  # get max delay for display purposes

# set parameters
Tmax = 150
dt = 0.0001
times = np.arange(0, Tmax + dt, dt)
T0 = 1.424555  # present day climatic temperature

# make array of investment
a_opt_it = np.zeros((len(df_secs.columns), len(times)))
a_d2_e_it = np.zeros((len(df_secs.columns), len(times)))
a_d2_i_it = np.zeros((len(df_secs.columns), len(times)))
a_d3_e_it = np.zeros((len(df_secs.columns), len(times)))
a_d3_i_it = np.zeros((len(df_secs.columns), len(times)))
a_d4_it = np.zeros((len(df_secs.columns), len(times)))

t0s_i = [0, ds3.delay_amount.values[-1], 0, 0, 0, 0, 0]
t0s_e = [0, 0, 0, 0, 0, ds3.delay_amount.values[-1], 0]
t0s_all = [ds4.delay_amount.values[-1]] * len(df_secs.columns)

for i in range(len(df_secs.columns)):
    a_opt_it[i] = get_abatement_path(ds2.opt_decarb_date.values[i], deltas[i],
                                     abars[i], r, ds2.optimal_carbon_price,
                                     cbars[i], Tmax=Tmax, dt=dt)
    a_d2_e_it[i] = get_abatement_path(ds2.decarb_date.values[delay_sec1,
                                                             delay_amount_ind,
                                                             i],
                                      deltas[i],
                                      abars[i], r,
                                      ds2.carbon_prices.values[delay_sec1,
                                                               delay_amount_ind,
                                                               i], cbars[i],
                                      Tmax=Tmax, dt=dt)
    a_d2_i_it[i] = get_abatement_path(ds2.decarb_date.values[delay_sec2,
                                                             delay_amount_ind,
                                                             i],
                                      deltas[i], abars[i], r,
                                      ds2.carbon_prices.values[delay_sec2,
                                                               delay_amount_ind,
                                                               i],
                                      cbars[i], Tmax=Tmax, dt=dt)

    a_d3_i_it[i] = get_abatement_path(ds3.decarb_date.values[delay_sec2,
                                                             delay_amount_ind,
                                                             i],
                                      deltas[i], abars[i], r,
                                      ds3.carbon_prices.values[delay_sec2,
                                                               delay_amount_ind,
                                                               i],
                                      cbars[i], t0=t0s_i[i], Tmax=Tmax, dt=dt)

    a_d3_e_it[i] = get_abatement_path(ds3.decarb_date.values[delay_sec1,
                                                             delay_amount_ind,
                                                             i],
                                      deltas[i], abars[i], r,
                                      ds3.carbon_prices.values[delay_sec1,
                                                               delay_amount_ind,
                                                               i],
                                      cbars[i], t0=t0s_e[i], Tmax=Tmax, dt=dt)

    a_d4_it[i] = get_abatement_path(ds4.decarb_date.values[delay_amount_ind,
                                                           i],
                                    deltas[i], abars[i], r,
                                    ds4.carbon_prices.values[delay_amount_ind,
                                                             i],
                                    cbars[i], t0=t0s_all[i], Tmax=Tmax, dt=dt)

# aggregate across sectors
total_emis = np.sum(abars)
total_a_opt = np.sum(a_opt_it, axis=0)
total_a_d2_e = np.sum(a_d2_e_it, axis=0)
total_a_d3_e = np.sum(a_d3_e_it, axis=0)
total_a_d4 = np.sum(a_d4_it, axis=0)

total_cum_opt = np.zeros_like(total_a_opt)
total_cum_d2 = np.zeros_like(total_a_opt)
total_cum_d3 = np.zeros_like(total_a_opt)
total_cum_d4 = np.zeros_like(total_a_opt)

tmp_cum_opt = 0
tmp_cum_d2 = 0
tmp_cum_d3 = 0
tmp_cum_d4 = 0
for t in range(1, len(times)):
    tmp_cum_opt += 0.0001 * (total_emis - total_a_opt[t])
    tmp_cum_d2 += 0.0001 * (total_emis - total_a_d2_e[t])
    tmp_cum_d3 += 0.0001 * (total_emis - total_a_d3_e[t])
    tmp_cum_d4 += 0.0001 * (total_emis - total_a_d4[t])

    total_cum_opt[t] = tmp_cum_opt
    total_cum_d2[t] = tmp_cum_d2
    total_cum_d3[t] = tmp_cum_d3
    total_cum_d4[t] = tmp_cum_d4

# print threshold crossings
b0_t = total_cum_opt * 0.44 * 1e-3 + T0
b1_t = total_cum_d2 * 0.44 * 1e-3 + T0
b2_t = total_cum_d3 * 0.44 * 1e-3 + T0
b3_t = total_cum_d4 * 0.44 * 1e-3 + T0

print("Years in optimal case where T < 1.6 deg C: {}".format(times[b0_t <=
                                                                   1.6][-1] + 2025))
print("Years in baseline 2 where T < 1.6 deg C: {}".format(times[b1_t <=
                                                                 1.6][-1] + 2025))
print("Years in baseline 3 where T < 1.6 deg C: {}".format(times[b2_t <=
                                                                 1.6][-1] + 2025))
print("Years in baesline 4 where T < 1.6 deg C: {}".format(times[b3_t <=
                                                                 1.6][-1] + 2025))


print("Years in optimal case where T >= 1.7 deg C: {}".format(times[b0_t >=
                                                                    1.699][0] + 2025))
print("Years in baseline 2 where T >= 1.7 deg C: {}".format(times[b1_t >=
                                                                  1.699][0] + 2025))
print("Years in baseline 3 where T >= 1.7 deg C: {}".format(times[b2_t >=
                                                                  1.699][0] + 2025))
print("Years in baesline 4 where T >= 1.7 deg C: {}".format(times[b3_t >=
                                                                  1.699][0] + 2025))
