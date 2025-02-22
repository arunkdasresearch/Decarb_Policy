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

from src.expost_paths import get_investment_path
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
x_opt_it = np.zeros((len(df_secs.columns), len(times)))
x_d2_e_it = np.zeros((len(df_secs.columns), len(times)))
x_d2_i_it = np.zeros((len(df_secs.columns), len(times)))
x_d3_e_it = np.zeros((len(df_secs.columns), len(times)))
x_d3_i_it = np.zeros((len(df_secs.columns), len(times)))
x_d4_it = np.zeros((len(df_secs.columns), len(times)))

t0s_i = [0, ds3.delay_amount.values[-1], 0, 0, 0, 0, 0]
t0s_e = [0, 0, 0, 0, 0, ds3.delay_amount.values[-1], 0]
t0s_all = [ds4.delay_amount.values[-1]] * len(df_secs.columns)

for i in range(len(df_secs.columns)):
    x_opt_it[i] = get_investment_path(ds2.opt_decarb_date.values[i], deltas[i],
                                      abars[i], r, ds2.optimal_carbon_price,
                                      cbars[i], Tmax=Tmax, dt=dt)
    x_d2_e_it[i] = get_investment_path(ds2.decarb_date.values[delay_sec1,
                                                              delay_amount_ind,
                                                              i],
                                       deltas[i],
                                       abars[i], r,
                                       ds2.carbon_prices.values[delay_sec1,
                                                                delay_amount_ind,
                                                                i], cbars[i],
                                       Tmax=Tmax, dt=dt)
    x_d2_i_it[i] = get_investment_path(ds2.decarb_date.values[delay_sec2,
                                                              delay_amount_ind,
                                                              i],
                                       deltas[i], abars[i], r,
                                       ds2.carbon_prices.values[delay_sec2,
                                                                delay_amount_ind,
                                                                i],
                                       cbars[i], Tmax=Tmax, dt=dt)

    x_d3_i_it[i] = get_investment_path(ds3.decarb_date.values[delay_sec2,
                                                              delay_amount_ind,
                                                              i],
                                       deltas[i], abars[i], r,
                                       ds3.carbon_prices.values[delay_sec2,
                                                                delay_amount_ind,
                                                                i],
                                       cbars[i], t0=t0s_i[i], Tmax=Tmax, dt=dt)

    x_d3_e_it[i] = get_investment_path(ds3.decarb_date.values[delay_sec1,
                                                              delay_amount_ind,
                                                              i],
                                       deltas[i], abars[i], r,
                                       ds3.carbon_prices.values[delay_sec1,
                                                                delay_amount_ind,
                                                                i],
                                       cbars[i], t0=t0s_e[i], Tmax=Tmax, dt=dt)

    x_d4_it[i] = get_investment_path(ds4.decarb_date.values[delay_amount_ind,
                                                            i],
                                     deltas[i], abars[i], r,
                                     ds4.carbon_prices.values[delay_amount_ind,
                                                              i],
                                     cbars[i], t0=t0s_all[i], Tmax=Tmax, dt=dt)

# insert nans at discontinuities to clean up plot
for i in range(len(df_secs.columns)):
    pos = np.where(np.abs(np.diff(x_d3_e_it[i])) >= 0.1)[0] + 1
    pos2 = np.where(np.abs(np.diff(x_d4_it[i])) >= 0.07)[0] + 1
    x_d3_e_it[i, pos] = np.nan
    x_d4_it[i, pos2] = np.nan

fig, ax = plt.subplots(2, 4, figsize=(20, 9.))

x_counter = 0
y_counter = 0

# define some aesthetic things
b4_style = 'dotted'
b4_color = color_list[-1]

b2_style = 'dashdot'

b3_style = 'solid'

labels = [['a)', 'b)', 'c)', 'd)'], ['e)', 'f)', 'g)', 'h)']]

for i in range(len(df_secs.columns)):
    # optimal
    ax[x_counter, y_counter].plot(times + 2025, cbars[i] * 0.5 *
                                  x_opt_it[i]**2,
                                  label='Policy suite 1:\nFirst-best',
                                  color='grey', alpha=0.8, linestyle='solid')

    # baseline 2
    ax[x_counter, y_counter].plot(times + 2025, cbars[i] * 0.5
                                  * x_d2_e_it[i]**2,
                                  linestyle=b2_style,
                                  color=color_list[delay_sec1],
                                  label='Policy suite 2:\nEnergy decarbonization\neffort relaxed')
    # ax[x_counter, y_counter].plot(times + 2020, cbars[i] * 0.5
    #                              * x_d2_i_it[i]**2,
    #                              label='Baseline 2: Industry\ndelayed by 10 yrs',
    #                              linestyle=b2_style,
    #                              color=color_list[delay_sec2])

    # baseline 3
    ax[x_counter, y_counter].plot(times + 2025, cbars[i] * 0.5
                                  * x_d3_e_it[i]**2,
                                  linestyle=b3_style,
                                  color=color_list[delay_sec1],
                                  label='Policy suite 3:\nEnergy decarbonization\neffort delayed')
    # ax[x_counter, y_counter].plot(times + 2020, cbars[i] * 0.5
    #                              * x_d3_i_it[i]**2,
    #                              label='Baseline 3: Industry\ndelayed by 10 yrs',
    #                              linestyle=b3_style,
    #                              color=color_list[delay_sec2])

    # baseline 4
    ax[x_counter, y_counter].plot(times + 2025, cbars[i] * 0.5
                                  * x_d4_it[i]**2,
                                  linestyle=b4_style,
                                  color=b4_color,
                                  label='Policy suite 4:\nEconomy-wide delay')

    ax[x_counter, y_counter].set_title(df_secs.columns.values[i])
    ax[x_counter, y_counter].set_xlim((2025, 2110))
    ax[x_counter, y_counter].set_xticks(ticks=[2025, 2050, 2075, 2100])

    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[x_counter, y_counter].text(0.9, 1.04, labels[x_counter][y_counter],
                                  transform=ax[x_counter, y_counter].transAxes
                                  + trans,
                                  fontsize=20, fontweight='bold',
                                  verticalalignment='top',
                                  bbox={'facecolor': 'none',
                                        'edgecolor': 'none', 'pad': 1})

    if i == 3:
        x_counter += 1
        y_counter = 0
    else:
        y_counter += 1

ax[0, 3].legend(bbox_to_anchor=(0.48, -1.26), loc='lower center')

# turn off 8th set of axes
ax[1, 3].set_visible(False)

# set y labels
ax[0, 0].set_ylabel("Investment effort\n(Billions of \$ yr$^{-1}$)")
ax[1, 0].set_ylabel("Investment effort\n(Billions of \$ yr$^{-1}$)")

# set x labels
for i in range(4):
    ax[1, i].set_xlabel("Year")

ax[0, -1].set_xlabel("Year")

for i in range(3):
    ax[0, i].set_xticklabels([])

fig.subplots_adjust(wspace=0.4, hspace=0.3)

if save_output:
    fig.savefig(basefile + cal + '_pfig4_investment_paths.png', dpi=400,
                bbox_inches='tight')
    print("Figure saved to: {}".format(basefile + cal +
                                       "_pfig4_investment_paths.png"))

else:
    plt.show()
