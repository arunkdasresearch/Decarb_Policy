"""Analysis script of delay impact on cost.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.16.2024

To run: python analysis_analytics.py [cal] [save_figs]
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import xarray as xr

from src.presets import get_presets

# parse command line input
cal = sys.argv[1]
save_figs = int(sys.argv[2])

# set presets
presets, basefile = get_presets()
plt.rcParams.update(presets)
plt.rcParams['text.usetex'] = True

# get data and open dataset
path = os.getcwd()
datapath = path + "/data/output/"

file_2 = cal + '_base2_analytics.nc'
file_3 = cal + '_base3_analytics.nc'
file_4 = cal + '_base4_analytics.nc'

ds2 = xr.open_dataset(datapath + file_2, engine='netcdf4')
ds3 = xr.open_dataset(datapath + file_3, engine='netcdf4')
ds4 = xr.open_dataset(datapath + file_4, engine='netcdf4')

# define color list from presets
color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
              '#0072B2', '#D55E00', '#CC79A7']

# make figure
# the plan is to have industry and energy be the poster-sectors, and plot the
# results for those sectors in baseline 2 and baseline 3
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# poster children indices
energy_ind = 5
industry_ind = 1

b4_style = 'dotted'
b4_label = 'Policy suite 4: Economy-wide delay'
b4_color = color_list[-1]

b3_style = 'solid'
b2_style = 'dashdot'

# now baseline 2 and 3 for energy and industry
for sec in [industry_ind, energy_ind]:
    # baseline 2
    ax[0].plot(ds2.delay_amount, ds2.cost[sec]/1000,
               label='Policy suite 2: ' + str(ds2.chal_sec[sec].values)
               + ' decarbonization effort relaxed',
               linestyle=b2_style, color=color_list[sec])
    ax[1].plot(ds2.delay_amount, ds2.cost[sec]/ds2.optimal_cost,
               label='Policy suite 2: ' + str(ds2.chal_sec[sec].values)
               + ' decarbonization effort relaxed',
               linestyle=b2_style,
               color=color_list[sec])
    ax[2].plot(ds2.delay_amount,
               np.gradient(ds2.cost[sec].values, ds2.delay_amount.values,
                           edge_order=2),
               label='Policy suite 2: ' + str(ds2.chal_sec[sec].values)
               + ' decarbonization effort relaxed',
               linestyle=b2_style, color=color_list[sec])

    # baseline 3
    ax[0].plot(ds3.delay_amount, ds3.cost[sec]/1000,
               label='Policy suite 3: ' + str(ds3.chal_sec[sec].values)
               + ' decarbonization effort delayed',
               linestyle=b3_style, color=color_list[sec])
    ax[1].plot(ds3.delay_amount, ds3.cost[sec]/ds3.optimal_cost,
               label='Policy suite 3: ' + str(ds3.chal_sec[sec].values)
               + ' decarbonization effort delayed',
               linestyle=b3_style,
               color=color_list[sec])
    ax[2].plot(ds3.delay_amount,
               np.gradient(ds3.cost[sec].values, ds3.delay_amount.values,
                           edge_order=2),
               label='Policy suite 3: ' + str(ds3.chal_sec[sec].values)
               + ' decarbonization effort delayed',
               linestyle=b3_style, color=color_list[sec])

# baseline 4
ax[0].plot(ds4.delay_amount, ds4.cost/1000,
           label=b4_label,
           linestyle=b4_style,
           color=b4_color)

ax[1].plot(ds4.delay_amount, ds4.cost/ds4.optimal_cost,
           label=b4_label,
           linestyle=b4_style,
           color=b4_color)

ax[2].plot(ds4.delay_amount,
           np.gradient(ds4.cost.values, ds4.delay_amount.values, edge_order=2),
           label=b4_label, linestyle=b4_style, color=b4_color)

labels = ['a)', 'b)', 'c)']
ltr = 0
for label in labels:
    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[ltr].text(0.05, 0.95, label,
                 transform=ax[ltr].transAxes + trans,
                 fontsize=20, fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(facecolor='none', edgecolor='none',
                           pad=1))
    ltr += 1


# axis labels and other aesthetics
ax[0].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")
ax[1].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")
ax[2].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")

ax[0].set_ylabel("Policy cost (Trillions of \$)")
ax[1].set_ylabel("Policy cost index (Efficient policy = 1)")
ax[2].set_ylabel("Marginal cost of delay (Billions of \$ yr$^{-1}$)")

ax[1].legend(bbox_to_anchor=(-1., -0.23), ncols=2, loc='upper left')

fig.subplots_adjust(wspace=0.25)

if save_figs:
    fig.savefig(basefile + cal + "_pfig6_aggcosts.png",
                dpi=400, bbox_inches='tight')
    print("Figure saved to:\n {}".format(basefile + cal + "_pfig6_aggcosts.png"))

else:
    plt.show()
