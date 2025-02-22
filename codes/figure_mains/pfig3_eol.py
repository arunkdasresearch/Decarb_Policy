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
from scipy.integrate import quad

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
inputpath = path + "/data/cal/" + cal + "_secs.csv"

file_2 = cal + '_base2_analytics.nc'
file_3 = cal + '_base3_analytics.nc'
file_4 = cal + '_base4_analytics.nc'

ds2 = xr.open_dataset(datapath + file_2, engine='netcdf4')
ds3 = xr.open_dataset(datapath + file_3, engine='netcdf4')
ds4 = xr.open_dataset(datapath + file_4, engine='netcdf4')

param_df = pd.read_csv(inputpath)

# define all the individual pieces


def valemis(t, T, mu, d, r=0.02, t0=0):
    multiplier = mu * np.exp(r * (t - t0) + d * t)
    integral = quad(lambda s: np.exp(-d * s), t, np.inf)
    if t > t0:
        val = multiplier * integral[0]
    elif t == t0:
        val = np.nan
    else:
        val = 0
    return val


def opp(t, T, mu, d, r=0.02, t0=0):
    multiplier = -mu * np.exp(r * (t - t0) + d * t)
    integral = quad(lambda s: np.exp(-d * s), T, np.inf)
    if t > t0:
        val = multiplier * integral[0]
    elif t == t0:
        val = np.nan
    else:
        val = 0
    return val


def longrun(t, T, mu, d, c, a, r=0.02, t0=0):
    val = c * a * d * np.exp(-(r + d) * (T - t))
    if t > t0:
        val = val
    elif t == t0:
        val = np.nan
    else:
        val = 0
    return val


# scenarios:
    # baseline 1
    # baseline 2: energy delayed by 10 years
    # baseline 3: energy delayed by 10 years
    # baseline 4
# index structure: (scenario, (E, O, E + O, L), sector, time)
dt = 0.1
tmax = 150
times = np.arange(0, tmax, dt)
bigdata = np.zeros((4, 4, 7, len(times)))

for i in range(7):
    if i == 5:
        t00 = 10.
        t01 = 10.
        t02 = 10.
        t03 = 10.
    elif i == 6:
        t00 = 0.
        t01 = 0.
        t02 = 0.
        t03 = 10.
    else:
        t00 = 0.
        t01 = 0.
        t02 = 0.
        t03 = 0

    sec = param_df.columns.values[i+1]

    # parse parameters
    tmp_d = param_df[sec].values[2]
    tmp_a = param_df[sec].values[1]
    tmp_c = param_df[sec].values[5]

    # we want the last decarbonization time of the current sector when energy
    # is delayed
    tmp_T1 = ds2.opt_decarb_date.values[i]
    tmp_T2 = ds2.decarb_date.values[5, -1, i]
    tmp_T3 = ds3.decarb_date.values[5, -1, i]
    tmp_T4 = ds4.decarb_date.values[-1, i]

    tmp_mu1 = ds2.optimal_carbon_price
    tmp_mu2 = ds2.carbon_prices.values[5, -1, i]
    tmp_mu3 = ds3.carbon_prices.values[5, -1, i]
    tmp_mu4 = ds4.carbon_prices.values[-1, i]

    for j in range(len(times)):
        if times[j] > tmp_T1:
            bigdata[0, 0, i, j] = np.nan
            bigdata[0, 1, i, j] = np.nan
            bigdata[0, 2, i, j] = np.nan
            bigdata[0, 3, i, j] = np.nan
        else:
            bigdata[0, 0, i, j] = valemis(times[j], tmp_T1, tmp_mu1, tmp_d)
            bigdata[0, 1, i, j] = opp(times[j], tmp_T1, tmp_mu1, tmp_d)
            bigdata[0, 2, i, j] = bigdata[0, 1, i, j] + bigdata[0, 0, i, j]
            bigdata[0, 3, i, j] = longrun(times[j], tmp_T1, tmp_mu1, tmp_d,
                                          tmp_c, tmp_a)

        if times[j] > tmp_T2:
            bigdata[1, 0, i, j] = np.nan
            bigdata[1, 1, i, j] = np.nan
            bigdata[1, 2, i, j] = np.nan
            bigdata[1, 3, i, j] = np.nan
        else:
            bigdata[1, 0, i, j] = valemis(times[j], tmp_T2, tmp_mu2, tmp_d)
            bigdata[1, 1, i, j] = opp(times[j], tmp_T2, tmp_mu2, tmp_d)
            bigdata[1, 2, i, j] = bigdata[1, 1, i, j] + bigdata[1, 0, i, j]
            bigdata[1, 3, i, j] = longrun(times[j], tmp_T2, tmp_mu2, tmp_d,
                                          tmp_c, tmp_a)

        if times[j] > tmp_T3:
            bigdata[2, 0, i, j] = np.nan
            bigdata[2, 1, i, j] = np.nan
            bigdata[2, 2, i, j] = np.nan
            bigdata[2, 3, i, j] = np.nan
        else:
            bigdata[2, 0, i, j] = valemis(times[j], tmp_T3, tmp_mu3, tmp_d,
                                          t0=t02)
            bigdata[2, 1, i, j] = opp(times[j], tmp_T3, tmp_mu3, tmp_d, t0=t02)
            bigdata[2, 2, i, j] = bigdata[2, 1, i, j] + bigdata[2, 0, i, j]
            bigdata[2, 3, i, j] = longrun(times[j], tmp_T3, tmp_mu3, tmp_d,
                                          tmp_c, tmp_a, t0=t02)

        if times[j] > tmp_T4:
            bigdata[3, 0, i, j] = np.nan
            bigdata[3, 1, i, j] = np.nan
            bigdata[3, 2, i, j] = np.nan
            bigdata[3, 3, i, j] = np.nan
        else:
            bigdata[3, 0, i, j] = valemis(times[j], tmp_T4, tmp_mu4, tmp_d,
                                          t0=t03)
            bigdata[3, 1, i, j] = opp(times[j], tmp_T4, tmp_mu4, tmp_d, t0=t03)
            bigdata[3, 2, i, j] = bigdata[3, 1, i, j] + bigdata[3, 0, i, j]
            bigdata[3, 3, i, j] = longrun(times[j], tmp_T4, tmp_mu4, tmp_d,
                                          tmp_c, tmp_a, t0=t03)


color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
              '#0072B2', '#D55E00', '#CC79A7']

fig, ax = plt.subplots(2, 4, figsize=(20, 10))

scenario_colors = ['grey', color_list[5], color_list[5], color_list[-1]]
scenario_ls = ['solid', 'dashdot', 'solid', 'dotted']
titles = ['Value of emissions\nreductions, $E$',
          'Forgone opportunity\neffect, $O$',
          '$E + O$', 'Long-term value of\nabatement capital, $L$']
baseline_label = ['Policy suite 1: First-best',
                  'Policy suite 2: Energy\ndecarbonization effort relaxed',
                  'Policy suite 3: Energy\ndecarbonization effort delayed',
                  'Policy suite 4: Economy-wide delay']

for j in [0, 1]:
    # ax[j, 0].set_ylabel(ylabels[j])
    for k in [0, 1, 2, 3]:
        ax[0, k].set_title(titles[k])
        for i in range(4):
            if j == 0:
                ax[j, k].plot(times + 2025, bigdata[i, k, -1],
                              color=scenario_colors[i],
                              linestyle=scenario_ls[i])
                ax[j, k].set_xticks([2025, 2050])

            else:
                ax[j, k].plot(times + 2025,
                              bigdata[i, k, 5],
                              color=scenario_colors[i],
                              linestyle=scenario_ls[i],
                              label=baseline_label[i])
                ax[j, k].set_xticks([2025, 2050])

ax[0, 0].set_ylabel("Buildings\n\nMarginal investment cost\n((\$  tCO$_2^{-1}$) / (GtCO$_2$ yr$^{-2}$))")
ax[1, 0].set_ylabel("Energy\n\nMarginal investment cost\n((\$  tCO$_2^{-1}$) / (GtCO$_2$ yr$^{-2}$))")

labels = [['a)', 'b)', 'c)', 'd)'], ['e)', 'f)', 'g)', 'h)']]
xtr = 0
ytr = 0
for i in range(8):
    trans = mtransforms.ScaledTranslation(0, 0, fig.dpi_scale_trans)
    ax[xtr, ytr].text(0.9, 1.04, labels[xtr][ytr],
                      transform=ax[xtr, ytr].transAxes + trans,
                      fontsize=22,
                      fontweight='bold', verticalalignment='top',
                      bbox=dict(facecolor='none', edgecolor='none', pad=1))
    ytr += 1
    if ytr == 4:
        ytr = 0
        xtr += 1

for i in range(4):
    ax[1, i].set_xlabel("Year")

ax[1, 2].legend(bbox_to_anchor=(-0.33, -0.4), loc='lower center', ncol=4)

# fig.tight_layout()
fig.subplots_adjust(wspace=0.3, hspace=0.3)

if save_figs:
    fig.savefig(basefile + cal + "_pfig3_EOL.png", dpi=400,
                bbox_inches='tight')
    print("Figure saved to:\n {}".format(basefile + cal + '_pfig3_EOL.png'))

else:
    plt.show()
