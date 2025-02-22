"""Analytic calculations for decarbonization dates and carbon prices.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.14.2024

To run: python analytic_calcs.py [cal] [save_output]
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from scipy.optimize import fsolve
from src.analytics import get_decarb_time_and_price, get_delay, total_cost, Bstar_pieces

# get rid of stupid future warnings stuff, can delete in the future if
# necessary (lol)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# parse command line input
cal = sys.argv[1]
save_output = int(sys.argv[2])

# get current working directory and make two paths to data
cwd = os.getcwd()
path_glob = ''.join([cwd, '/data/cal/', cal, '_glob.csv'])
path_secs = ''.join([cwd, '/data/cal/', cal, '_secs.csv'])

# import csvs using Pandas
df_glob = pd.read_csv(path_glob, delimiter=',', header=0)
df_secs = pd.read_csv(path_secs, delimiter=',', header=0, index_col=0)

# make list of model parameters
# note the last 0 is for the delay in starting doing anything, i.e., it is t0
params = []
for col in df_secs.columns:
    params.append([df_secs[col]['delta'], df_secs[col]['abar'],
                   df_secs[col]['cbar'], 0])
params = np.array(params)

# find optimal decarbonization dates and carbon price
optimal = fsolve(get_decarb_time_and_price, x0=(20, 50, 40, 40, 40, 40, 40,
                                                50), args=[df_glob['r'].values,
                                                           df_glob['B'].values,
                                                           params])
# find optimal cost of policy
opt_cost = total_cost(optimal[:-1], [optimal[-1]] * 7, params, df_glob['r'], 0)

# find optimal emissions allocations for each sector
bstars = []
for i in range(7):
    T = optimal[i]

    # find Bstar pieces
    tmp_bstar_1, tmp_bstar_2 = Bstar_pieces(T, df_glob['r'].values, params[i])

    # Bstar = bstar_noprice + price * bstar_price, where price is optimal
    # carbon price
    tmp_bstar = tmp_bstar_1 + optimal[-1] * tmp_bstar_2
    bstars.append(tmp_bstar[0])

# make new parameter list with optimal emissions allocation included
params_w_bstar = []
for i in range(7):
    col = df_secs.columns[i]
    params_w_bstar.append([df_secs[col]['delta'], df_secs[col]['abar'],
                           df_secs[col]['cbar'], 0, bstars[i]])
params_w_bstar = np.array(params_w_bstar)

# delay sectors indices and delay amounts
delay_inds = np.arange(0, 7, 1)

delay_amounts = np.arange(0.1, 10.1, 0.1)  # in years of delay

# big arrays of costs, emissions premiums, carbon prices, and decarbonization
# dates to store our results in
big_costs = np.zeros((len(delay_inds), len(delay_amounts)))
sec_costs = np.zeros((len(delay_inds), len(delay_amounts),
                      len(df_secs.columns)))
B_ps = np.zeros((len(delay_inds), len(delay_amounts)))
big_prices = np.zeros((len(delay_inds), len(delay_amounts),
                       len(df_secs.columns)))
big_Ts = np.zeros((len(delay_inds), len(delay_amounts), len(df_secs.columns)))

# counter for delays and sectors
delay_count = 0
delay_sec_count = 0

# loop through sectors and delay amounts and compute cost, decarbonization
# dates, and carbon prices with delay
for delay_ind in delay_inds:
    print("\nDoing {} sector calculations...".format(df_secs.columns[delay_ind]))
    costs = []
    for delay_amount in delay_amounts:
        print("Delay amount: {}".format(delay_amount))
        # compute the amount of emissions required for delay_amount worth of
        # delay
        delay = fsolve(get_delay, x0=(25), args=[delay_amount, delay_ind,
                                                 df_glob['r'], params_w_bstar])

        # extrat emissions amount
        B_p_star = delay[0]

        # store in big list for saving later
        B_ps[delay_sec_count, delay_count] = B_p_star

        # compute the decarbonization time and price for the challenged sector
        Tc, mu_c = fsolve(get_decarb_time_and_price,
                          x0=(optimal[delay_ind], optimal[-1]),
                          args=[df_glob['r'], params_w_bstar[delay_ind, -1] + B_p_star, [params_w_bstar[delay_ind, :-1]]])

        # compute the decarbonization dates and price for the expedient sectors
        guess = np.delete(optimal, delay_ind)
        opt_e = fsolve(get_decarb_time_and_price,
                       x0=guess, 
                       args=[df_glob['r'], np.sum(np.delete(params_w_bstar,
                                                            delay_ind,
                                                            axis=0).T[-1]) -
                             B_p_star, np.delete(params_w_bstar, delay_ind,
                                                 axis=0)[:, :-1]])

        # store decarbonization times and carbon prices for each sector
        Ts_const = np.zeros(len(df_secs.columns))
        prices_const = np.zeros(len(df_secs.columns))
        Ts_const[delay_ind] = Tc
        prices_const[delay_ind] = mu_c

        Ts_const[:delay_ind] = opt_e[:delay_ind]
        Ts_const[(delay_ind+1):] = opt_e[delay_ind:-1]

        prices_const[:delay_ind] = opt_e[-1]
        prices_const[(delay_ind+1):] = opt_e[-1]

        # compute the total cost of the this policy
        costs = total_cost(Ts_const, prices_const, params, df_glob['r'],
                           delay_ind)
        costtotal = np.sum(costs)

        # store output
        big_costs[delay_sec_count, delay_count] = costtotal
        sec_costs[delay_sec_count, delay_count] = costs
        big_Ts[delay_sec_count, delay_count] = Ts_const
        big_prices[delay_sec_count, delay_count] = prices_const

        # index the delay count
        delay_count += 1

    # add delay sector counter
    delay_sec_count += 1

    # reset the delay counter
    delay_count = 0

    print("{} sector calculations complete.".format(df_secs.columns[delay_ind]))

# make big dataset of results
ds = xr.Dataset(data_vars={'cost': (['chal_sec', 'delay_amount'], big_costs),
                           'Bp': (['chal_sec', 'delay_amount'], B_ps),
                           'carbon_prices': (['chal_sec', 'delay_amount', 'sector'], big_prices),
                           'decarb_date': (['chal_sec', 'delay_amount', 'sector'], big_Ts),
                           'opt_costs': (['sector'], opt_cost),
                           'opt_decarb_date': (['sector'], optimal[:-1]),
                           'sec_costs': (['chal_sec', 'delay_amount',
                                          'sector'], sec_costs)},
                coords={'chal_sec': (['chal_sec'], df_secs.columns),
                        'delay_amount': (['delay_amount'], delay_amounts),
                        'sector': (['sector'], df_secs.columns)},
                attrs={'optimal_carbon_price': optimal[-1],
                       'optimal_cost': np.sum(opt_cost)})

if save_output:
    # get current working directory, save output to data/output/ folder
    cwd = os.getcwd()
    path = ''.join([cwd, '/data/output/', cal, '_' + 'base2_analytics.nc'])
    ds.to_netcdf(path=path, mode='w', format='NETCDF4', engine='netcdf4')
    print("\n------------------------------------------")
    print("Data successfully saved to file:\n{}".format(path))
    print("------------------------------------------\n")

else:
    print("\n-------------------------------------------------")
    print("Model exited optimally. Printing results below.")
    print("-------------------------------------------------\n")
    print(ds)
