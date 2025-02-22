"""Analysis script of delay impact on cost.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.16.2024

To run: python cost_param.py [cal] [save_figs]
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.transforms as mtransforms

from src.presets import get_presets
from sklearn.linear_model import LinearRegression

# parse command line input
cal = sys.argv[1]
save_figs = int(sys.argv[2])

# set presets
presets, basefile = get_presets()
plt.rcParams.update(presets)
plt.rcParams['text.usetex'] = True

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

delay_ind_1 = 9
delay_ind_5 = 29
delay_ind_10 = -1

del_ind = 2
abar_ind = 1
cbar_ind = 5

fig, ax = plt.subplots(2, 3, figsize=(14, 9), sharey='row', sharex='col')

color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
              '#0072B2', '#D55E00', '#CC79A7']

for sec in range(len(ds2.chal_sec.values)):
    sec_param = param_df.columns.values[sec+1]
    ax[0, 0].scatter(param_df[sec_param].values[del_ind] * 100,
                     ds2.cost[sec, delay_ind_5] / ds2.optimal_cost,
                     label=ds2.chal_sec[sec].values, marker='+',
                     color=color_list[sec])

    ax[0, 1].scatter(param_df[sec_param].values[abar_ind],
                     ds2.cost[sec, delay_ind_5] / ds2.optimal_cost,
                     label=ds2.chal_sec[sec].values, marker='+',
                     color=color_list[sec])

    ax[0, 2].scatter(np.log10(param_df[sec_param].values[cbar_ind]),
                     ds2.cost[sec, delay_ind_5]/ds2.optimal_cost,
                     marker='+', color=color_list[sec])

    ax[1, 0].scatter(param_df[sec_param].values[del_ind] * 100,
                     ds3.cost[sec, delay_ind_5]/ds3.optimal_cost,
                     label=ds3.chal_sec[sec].values, marker='+',
                     color=color_list[sec])

    ax[1, 1].scatter(param_df[sec_param].values[abar_ind],
                     ds3.cost[sec, delay_ind_5]/ds3.optimal_cost,
                     label=ds3.chal_sec[sec].values, marker='+',
                     color=color_list[sec])

    ax[1, 2].scatter(np.log10(param_df[sec_param].values[cbar_ind]),
                     ds3.cost[sec, delay_ind_5]/ds3.optimal_cost,
                     marker='+', color=color_list[sec])

for sec in range(len(ds2.chal_sec.values)):
    sec_param = param_df.columns.values[sec+1]
    ax[0, 0].scatter(param_df[sec_param].values[del_ind] * 100,
                     ds2.cost[sec, delay_ind_10]/ds2.optimal_cost,
                     label=ds2.chal_sec[sec].values, marker='o',
                     color=color_list[sec])

    ax[0, 1].scatter(param_df[sec_param].values[abar_ind],
                     ds2.cost[sec, delay_ind_10]/ds2.optimal_cost,
                     label=ds2.chal_sec[sec].values, marker='o',
                     color=color_list[sec])

    ax[0, 2].scatter(np.log10(param_df[sec_param].values[cbar_ind]),
                     ds2.cost[sec, delay_ind_10]/ds2.optimal_cost,
                     label=ds2.chal_sec[sec].values, marker='o',
                     color=color_list[sec])

    ax[1, 0].scatter(param_df[sec_param].values[del_ind] * 100,
                     ds3.cost[sec, delay_ind_10]/ds3.optimal_cost,
                     label=ds3.chal_sec[sec].values, marker='o',
                     color=color_list[sec])

    ax[1, 1].scatter(param_df[sec_param].values[abar_ind],
                     ds3.cost[sec, delay_ind_10]/ds3.optimal_cost,
                     label=ds3.chal_sec[sec].values, marker='o',
                     color=color_list[sec])

    ax[1, 2].scatter(np.log10(param_df[sec_param].values[cbar_ind]),
                     ds3.cost[sec, delay_ind_10]/ds3.optimal_cost,
                     label=ds3.chal_sec[sec].values, marker='o',
                     color=color_list[sec])

for i in range(3):
    ax[0, i].set_ylim((0.9995, 1.021))
    ax[0, i].set_yticks([1.0, 1.005, 1.01, 1.015, 1.02])
    ax[0, i].set_yticklabels([1, 1.005, 1.01, 1.015, 1.02])

ax[0, 0].set_title("Policy suite 2: Sector decarbonization effort relaxed")
ax[1, 0].set_title("Policy suite 3: Sector decarbonization effort delayed")

ax[0, 2].scatter([-1e10], [0], label='10 year delay', color='grey', marker='o')
ax[0, 2].scatter([-1e10], [0], label='5 year delay', color='grey', marker='+')
ax[0, 2].set_xlim((2.8, 4.2))
ax[1, 0].set_xlabel(r"Captial depreciation rate," + "\n"
                    + r"$\delta_i$ (\% / yr)")
ax[1, 1].set_xlabel("Abatement potential," + "\n"
                    + r"$\bar{a}_i$ (GtCO$_2$ / yr)")
ax[1, 2].set_xlabel(r"Log marginal investment cost," + "\n"
                    + r"$\log_{10}(\bar{c}_i)$ (\$ / (tCO$_2$/yr$^2$))")

ax[0, 0].set_ylabel("Relative policy cost\n(1 = Optimal policy)")
ax[1, 0].set_ylabel("Relative policy cost\n(1 = Optimal policy)")

ax[0, 2].legend(bbox_to_anchor=(1.05, 1.0), ncols=1, loc='upper left')

trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)

labels = [['a)', 'b)', 'c)'], ['d)', 'e)', 'f)']]

x_pos = [[0.6, 0.07, 0.6], [0.6, 0.07, 0.6]]
y_pos = [[0.87, 0.96, 0.87], [0.87, 0.96, 0.87]]
diff = 0.12

xtr = 0
ytr = 0
for i in range(6):
    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[xtr, ytr].text(0.9, 0.96, labels[xtr][ytr],
                      transform=ax[xtr, ytr].transAxes + trans,
                      fontsize=22,
                      fontweight='bold', verticalalignment='top',
                      bbox=dict(facecolor='none', edgecolor='none', pad=1))

    ytr += 1
    if ytr == 3:
        ytr = 0
        xtr += 1

fig.tight_layout()

if save_figs:
    fig.savefig(basefile + cal + "_pfig7_cost_params.png", dpi=400,
                bbox_inches='tight')
    print("Figure saved to:\n {}".format(basefile + cal +
                                         "_pfig7_cost_params.png"))

else:
    plt.show()
