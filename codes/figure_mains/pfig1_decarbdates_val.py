"""Plot paths resulting from numerical simulations of decarbonization dates.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.28.2024

To run: python paths.py [cal] [save_output]
"""

import os
import sys

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from src.presets import get_presets

# ignore future warnings :)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# choose interesting cases
delay_sec1 = 5  # energy
delay_sec2 = 1  # industry
delay_amount_ind = -1  # get max delay for display purposes

# make figure
fig, ax = plt.subplots(2, 4, sharey=True, figsize=(18, 9))

# define some aesthetic things
b4_label = 'Policy suite 4:\nEconomy-wide delay'
b4_style = 'dotted'
b4_color = color_list[-1]

b2_label = 'Policy suite 2:\n'
b2_style = 'dashdot'

b3_label = 'Policy suite 3:\n'
b3_style = 'solid'

x_counter = 0
y_counter = 0
labels = [['a)', 'b)', 'c)', 'd)'], ['e)', 'f)', 'g)', 'h)']]

for i in range(len(ds4.sector.values)):
    # baseline 2: energy
    ax[x_counter, y_counter].plot(ds2.delay_amount,
                                  ds2.decarb_date.values[delay_sec1, :, i]
                                  - ds2.opt_decarb_date.values[i],
                                  label=b2_label
                                  + str(ds2.sector.values[delay_sec1])
                                  + ' decarbonization\neffort relaxed',
                                  linestyle=b2_style,
                                  color=color_list[delay_sec1])

    # baseline 2: energy
    ax[x_counter, y_counter].plot(ds2.delay_amount,
                                  ds2.decarb_date.values[delay_sec2, :, i]
                                  - ds2.opt_decarb_date.values[i],
                                  label=b2_label
                                  + str(ds2.sector.values[delay_sec2])
                                  + ' decarbonization\neffort relaxed',
                                  linestyle=b2_style,
                                  color=color_list[delay_sec2])
    # baseline 3: energy
    ax[x_counter, y_counter].plot(ds3.delay_amount,
                                  ds3.decarb_date.values[delay_sec1, :, i]
                                  - ds3.opt_decarb_date.values[i],
                                  label=b3_label
                                  + str(ds3.sector.values[delay_sec1])
                                  + ' decarbonization\neffort delayed',
                                  linestyle=b3_style,
                                  color=color_list[delay_sec1])

    # baseline 3: industry
    ax[x_counter, y_counter].plot(ds3.delay_amount,
                                  ds3.decarb_date.values[delay_sec2, :, i]
                                  - ds3.opt_decarb_date.values[i],
                                  label=b3_label
                                  + str(ds3.sector.values[delay_sec2])
                                  + ' decarbonization\neffort delayed',
                                  linestyle=b3_style,
                                  color=color_list[delay_sec2])

    # baseline 4
    ax[x_counter, y_counter].plot(ds4.delay_amount,
                                  ds4.decarb_date.values[:, i]
                                  - ds4.opt_decarb_date.values[i],
                                  label=b4_label,
                                  linestyle=b4_style,
                                  color=b4_color)

    ax[x_counter, y_counter].set_title(ds4.sector.values[i])
    ax[x_counter, y_counter].set_ylim((-45, 0))

    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[x_counter, y_counter].text(0.05, 0.14, labels[x_counter][y_counter],
                                  transform=ax[x_counter, y_counter].transAxes
                                  + trans,
                                  fontsize=20, fontweight='bold',
                                  verticalalignment='top',
                                  bbox=dict(facecolor='none', edgecolor='none',
                                            pad=1))

    if i == 3:
        x_counter += 1
        y_counter = 0
    else:
        y_counter += 1

ax[0, 3].legend(bbox_to_anchor=(0.5, -1.6), loc='lower center',
                fontsize=14)

# turn off 8th set of axes
ax[1, 3].set_visible(False)

# set y labels
ax[0, 0].set_ylabel("Change in decarbonization\ndate (yrs)")
ax[1, 0].set_ylabel("Change in decarbonization\ndate (yrs)")

# set x labels
for i in range(4):
    ax[1, i].set_xlabel('Delay in decarbonizing\nchallenged sector (yrs)')

ax[0, -1].set_xlabel("Delay in decarbonizing\nchallenged sector (yrs)")

for i in range(3):
    ax[0, i].set_xticklabels([])
    ax[1, i].set_xticks([0, 5, 10])
    ax[1, i].set_xticklabels([0, 5, 10])

ax[0, 3].set_xticks([0, 5, 10])
ax[0, 3].set_xticklabels([0, 5, 10])

fig.subplots_adjust(wspace=0.4, hspace=0.4)

if save_output:
    fig.savefig(basefile + cal + '_pfig1_decarb_dates_val.png', dpi=400,
                bbox_inches='tight')
    print("Figure saved to: {}".format(basefile + cal +
                                       '_pfig1_decarb_dates_val.png'))

else:
    plt.show()
