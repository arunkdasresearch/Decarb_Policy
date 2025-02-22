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
fig, ax = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)

# poster children indices
energy_ind = 5
industry_ind = 1

b4_style = 'dotted'
b4_color = color_list[-1]

non_style = 'dashed'
chal_style = 'solid'

# now baseline 2 and 3 for energy and industry
for sec in [industry_ind, energy_ind]:
    # baseline 2
    ax[0].plot(ds2.delay_amount,
               ds2.carbon_prices[sec, :, sec+1]/ds2.optimal_carbon_price,
               label=str(ds2.chal_sec[sec].values)
               + ' relaxed,\nnon-challenged sectors',
               linestyle=non_style, color=color_list[sec])
    ax[0].plot(ds2.delay_amount,
               ds2.carbon_prices[sec, :, sec]/ds2.optimal_carbon_price,
               label=str(ds2.chal_sec[sec].values)
               + ' relaxed,\nchallenged sector',
               linestyle=chal_style, color=color_list[sec])

    # baseline 3
    ax[1].plot(ds3.delay_amount,
               ds3.carbon_prices[sec, :, sec+1]
               / ds3.optimal_carbon_price,
               label=str(ds3.chal_sec[sec].values)
               + ' delayed, non-challenged sectors',
               linestyle=non_style, color=color_list[sec])
    ax[1].plot(ds3.delay_amount,
               ds3.carbon_prices[sec, :, sec]
               # * np.exp(-0.02 * ds3.delay_amount.values)
               / ds3.optimal_carbon_price,
               label=str(ds3.chal_sec[sec].values)
               + ' delayed, challenged sector',
               linestyle=chal_style, color=color_list[sec])

# baseline 4
ax[2].plot(ds4.delay_amount,
           ds4.carbon_prices[:, 0]
           / ds4.optimal_carbon_price,
           linestyle=b4_style,
           label="All sectors",
           color=b4_color)

# panel labels
labels = ['a)', 'b)', 'c)']
ltr = 0
for label in labels:
    if label == 'a)' or label == 'b)':
        xloc = 0.9
    else:
        xloc = 0.85
    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[ltr].text(xloc, 0.95, label,
                 transform=ax[ltr].transAxes + trans,
                 fontsize=20, fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(facecolor='none', edgecolor='none',
                           pad=1))
    ltr += 1
# set panel titles
ax[0].set_title("Policy suite 2: Decarbonization effort\nof challenged sector relaxed",
                fontsize=18)
ax[1].set_title("Policy suite 3: Decarbonization effort\nin challenged sector delayed",
                fontsize=18)
ax[2].set_title("Policy suite 4: Economy-wide delay", fontsize=18)

# axis labels and other aesthetics
ax[0].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")
ax[1].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")
ax[2].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")

ax[0].set_ylabel("Carbon price index\n(First-best policy = 1)")

ax[0].legend(ncols=1, loc='upper left')
ax[2].legend(loc='upper left')

fig.subplots_adjust(wspace=0.25)

if save_figs:
    fig.savefig(basefile + cal + "_pfig2_carbonprices.png",
                dpi=400, bbox_inches='tight')
    print("Figure saved to:\n {}".format(basefile + cal +
                                         '_pfig2_carbonprices.png'))

else:
    plt.show()
