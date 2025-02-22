"""Analysis script of delay impact on cost.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.16.2024

To run: python analysis_analytics.py [cal] [save_figs]
"""

import os
import sys

import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.transforms as mtransforms

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
fig, ax = plt.subplots(2, 4, figsize=(20, 10), sharey=True)

# poster children indices
energy_ind = 5
industry_ind = 1

b4_style = 'dotted'
b4_label = 'Policy suite 4:\nEconomy-wide delay'
b4_color = color_list[-1]

b3_style = 'solid'
b2_style = 'dashdot'

# define trackers
xtr = 0
ytr = 0
s_tr = 0

# set panel labels
labels = [['a)', 'b)', 'c)', 'd)'], ['e)', 'f)', 'g)', 'h)']]

for p in range(7):
    for sec in [energy_ind]:
        # baseline 2
        ax[xtr, ytr].plot(ds2.delay_amount,
                          ds2.sec_costs[sec, :, s_tr]/ds2.opt_costs[s_tr],
                          label='Policy suite 2:\n' + str(ds2.chal_sec[sec].values)
                          + ' decarbonization\neffort relaxed',
                          linestyle=b2_style,
                          color=color_list[sec])

        # baseline 3
        ax[xtr, ytr].plot(ds3.delay_amount,
                          ds3.sec_costs[sec, :, s_tr]/ds3.opt_costs[s_tr],
                          label='Policy suite 3:\n' + str(ds3.chal_sec[sec].values)
                          + ' decarbonization\neffort delayed',
                          linestyle=b3_style,
                          color=color_list[sec])

    # baseline 4
    ax[xtr, ytr].plot(ds4.delay_amount,
                      ds4.sec_costs[:, s_tr]/ds4.opt_costs[s_tr],
                      label=b4_label,
                      linestyle=b4_style,
                      color=b4_color)

    # set title and other aesthetics
    ax[xtr, ytr].set_title(ds2.sector.values[s_tr])
    ax[xtr, ytr].axhline(1.0,
                         0, max(ds4.delay_amount.values),
                         linestyle='solid',
                         color='black',
                         linewidth=1.)
    ax[xtr, ytr].set_xlim(-0.1, 10.1)

    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[xtr, ytr].text(0.05, 0.96, labels[xtr][ytr],
                      transform=ax[xtr, ytr].transAxes
                      + trans,
                      fontsize=20, fontweight='bold',
                      verticalalignment='top',
                      bbox=dict(facecolor='none', edgecolor='none',
                                pad=1))
    if xtr == 1:
        ax[xtr, ytr].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")

    if ytr == 0:
        ax[xtr, ytr].set_ylabel("Policy cost index\n(1 = Efficient policy)")

    if xtr == 0 and ytr < 3:
        ax[xtr, ytr].set_xticklabels([])

    if xtr == 0 and ytr == 3:
        ax[xtr, ytr].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")

    # index trackers
    s_tr += 1
    ytr += 1

    if ytr == 4:
        ytr = 0
        xtr = 1


ax[1, 3].set_visible(False)

ax[0, 3].legend(bbox_to_anchor=(0.49, -1.05), loc='lower center')

fig.subplots_adjust(wspace=0.25)

if save_figs:
    fig.savefig(basefile + cal + "_pfig5_seccosts.png",
                dpi=400, bbox_inches='tight')
    print("Figure saved to:\n {}".format(basefile + cal + "_pfig5_seccosts.png"))

else:
    plt.show()
