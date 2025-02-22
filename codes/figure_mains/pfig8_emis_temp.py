"""Plot paths resulting from numerical simulations of decarbonization dates.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.28.2024

To run: python paths.py [cal] [save_output]
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from src.expost_paths import get_abatement_path
from src.presets import get_presets

# parse command line
cal = sys.argv[1]
save_output = int(sys.argv[2])

# set presets for plotting
presets, basefile = get_presets()
plt.rcParams.update(presets)
plt.rcParams['text.usetex'] = True

# need color list for this one
color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
              '#0072B2', '#D55E00', '#CC79A7']

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

fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))

# define some aesthetic things
b4_style = 'dotted'
b4_color = color_list[-1]

b2_style = 'dashdot'
b3_style = 'solid'

# plot emissions
ax[0].plot(times + 2024, total_emis - total_a_opt,
           label='Policy suite 1:\nFirst-best policy',
           color='grey', alpha=0.8, linestyle='solid')

# baseline 2
ax[0].plot(times + 2025,
           total_emis - total_a_d2_e,
           linestyle=b2_style,
           color=color_list[delay_sec1],
           label='Policy suite 2: Energy\ndecarbonization effort relaxed')

# baseline 3
ax[0].plot(times + 2025, total_emis - total_a_d3_e,
           linestyle=b3_style,
           color=color_list[delay_sec1],
           label='Policy suite 3: Energy\ndecarbonization effort delayed')

# baseline 4
ax[0].plot(times + 2025,
           total_emis - total_a_d4,
           linestyle=b4_style,
           color=b4_color,
           label='Policy suite 4: Economy-wide\ndelay')

# plot implied temperature change
T0 = 1.424555
ax[1].plot(times + 2025, total_cum_opt * 0.44 * 1e-3 + T0,
           color='grey', alpha=0.8, linestyle='solid')

# baseline 2
ax[1].plot(times + 2025,
           total_cum_d2 * 0.44 * 1e-3 + T0,
           linestyle=b2_style,
           color=color_list[delay_sec1])

# baseline 3
ax[1].plot(times + 2025, total_cum_d3 * 0.44 * 1e-3 + T0,
           linestyle=b3_style,
           color=color_list[delay_sec1])

# baseline 4
ax[1].plot(times + 2025,
           total_cum_d4 * 0.44 * 1e-3 + T0,
           linestyle=b4_style,
           color=b4_color)

ax[1].axhline(1.7, 0, 2200, linewidth=2,
              linestyle='dashed', color=color_list[3], zorder=1,
              label="Temperature target")

labels = ['a)', 'b)']
for i in range(2):
    ax[i].set_xlim((2025, 2110))
    ax[i].set_xticks(ticks=[2025, 2050, 2075, 2100])
    # ax[i].set_xticks(ticks=[2025, 2040, 2060, 2080, 2100, 2120],
    #                  minor=True)
    # ax[i].tick_params(axis='x', which='minor', length=6, width=1)

    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[i].text(0.9, 0.92, labels[i],
               transform=ax[i].transAxes
               + trans,
               fontsize=20, fontweight='bold',
               verticalalignment='top',
               bbox={'facecolor': 'none',
                     'edgecolor': 'none', 'pad': 1})

ax[0].legend(loc='upper right', fontsize=14)
ax[1].legend(loc='lower right', fontsize=14)

# set y labels
ax[0].set_ylabel("Emissions (GtCO$_2$ yr$^{-1}$)")
ax[1].set_ylabel("Global average temperature above\npreindustrial ($^\circ$C)")

# set x labels
for i in range(2):
    ax[i].set_xlabel("Year")

fig.tight_layout()

if save_output:
    fig.savefig(basefile + cal + '_pfig8_emis_temp.png', dpi=400,
                bbox_inches='tight')
    print("Figure saved to: {}".format(basefile + cal + "_pfig8_emis_temp.png"))

else:
    x = 9
    #plt.show()
