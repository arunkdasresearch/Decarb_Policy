"""Analytic functions required for calculations on delayed decarbonization.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group

5.16.2024
"""

import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import quad


def aT(T, price, r, params):
    """Compute the abatement at the decarbonization date.

    Parameters
    ----------
    T: float
        the decarbonization time in years
    
    price: float
        the carbon price

    r: float
        the social discount rate

    params: (3,) list
        list of sector-specific parameters
        [captial depreciation rate, emissions rate, marginal investment cost]

    Returns
    -------
    total: float
        the total abatement at time T
    """

    # unpack parameters
    delta, alpha, c, t0 = params

    # this equation comes in three pieces
    # first piece
    aT1 = price * np.exp(-r * t0) * (np.exp(T * (r + delta)) - np.exp(t0 * (r +
                                                                           delta)))
    aT1 /= c * (r + delta)

    # second piece
    aT2 = price * (np.exp((2 * t0 - T) * delta) - np.exp(-r * t0 + T * (r +
                                                                        delta)))
    aT2 /= c * (r + 2 * delta)

    # third piece
    aT3 = np.exp(T * (r + 2 * delta)) - np.exp(t0 * (r + 2 * delta))
    aT3 *= alpha * delta**2 * np.exp(-T * (r + delta))
    aT3 /= r + 2 * delta

    # sum pieces and make final adjustments
    aT = aT1 + aT2 + aT3
    aT *= np.exp(-delta * T)
    aT /= delta

    # return final product!
    return aT

def Bstar_pieces(T, r, params):
    """Compute the optimal sectoral allocation of emissions pieces.

    The optimal sectoral allocation of emissions, Bstar, can be represented as:

    Bstar = A + carbon_price * B

    This function returns A and B.

    Parameters
    ----------
    T: float
        decarbonization date

    r: float
        social discount rate
    
    params: (3,) list
        list of sector-specific parameters
        [captial depreciation rate, emissions rate, marginal investment cost]
    
    Returns
    -------
    total_noprice: float
        A in the above; the piece of Bstar that isn't affected by the carbon price
    
    total_price: float
        B in the above; the piece of Bstar that is affected by the carbon price
    """

    # unpack parameters
    delta, alpha, c, t0 = params
    dt = t0 - T

    # piece not connected to the carbon price
    noprice_piece = - delta + np.exp(dt * (r + delta)) * delta\
            + (r + delta) * np.exp(dt * (r + delta))\
            - (r + delta) * np.exp(dt * (r + 2 * delta))\
            - dt * (r + 2 * delta) * (r + delta)

    noprice_piece *= alpha / ((r + delta) * (r + 2 * delta))

    # piece connected to the carbon price
    price_piece = r**2 * (np.exp(dt * delta) - 1)**2\
            + r * delta * (3 - 4 * np.exp(dt * delta)
                           + np.exp(2 * dt * delta))\
            - 2 * delta**2 * (np.exp(-r * dt) - 1)

    price_piece /= r * c * delta**2 * (r + delta) * (r + 2 * delta)

    # return both pieces
    return noprice_piece, price_piece

def cprice(Ts, B, r, all_params):
    """Compute the carbon price for a set of sectors.

    Parameters
    ----------
    Ts: (N_secs,) list
        list of final decarbonization dates

    B: float
        total cap for sectors

    r: float
        social discount rate

    all_params: (N_secs, 3) list
        parameters for each sector
        [capital depreciation rate, emissions rate, marginal investment cost]

    Returns
    -------
    cprice: float
        the carbon price
    """

    # number of sectors in this grouping
    N_secs = np.shape(all_params)[0]

    # initalize the list
    # see the Mathematica notebook or paper for this equation and the rationale
    ## cprice = (B - sum[noprice_Bstar_pieces])/sum[price_Bstar_piece]
    sum_first = 0
    sum_second = 0
    for i in range(N_secs):
        tmp_first, tmp_second = Bstar_pieces(Ts[i], r, all_params[i])
        sum_first += tmp_first
        sum_second += tmp_second

    # compute carbon price
    cprice = (B - sum_first) / sum_second 
    return cprice

def get_decarb_time_and_price(control, params):
    """Get decarbonization times and carbon prices for a set of sectors.

    This function is meant to be an input to scipy.optimize.fsolve. It returns a 
    set of decarbonization dates for each sector, as well as the overall carbon price.

    Parameters
    ----------
    control: (N_secs + 1,) list
        The decarbonization dates for N_secs, plus the carbon price

    params: list
        parameters needed for the optimization
        - params[0]: social discount rate, r
        - params[1]: total cap, B
        - params[2]: (N_secs, 3) array of sectoral parameters

    Returns
    -------
    final_list: (N_secs + 1,) list
        list of differences between control and model solutions
        - final_list[:-1]: difference between abatement at time T and emissions rate
                           should be zero when the proper set of Ts are calculated
        - final_list[-1]: the difference between the carbon price in the optimization and
                          the computed carbon price implied by the Ts
                          should be zero when the proper set of Ts are calculated
    """
    
    # parse control input
    Ts = control[:-1]
    price = control[-1]

    # parse parameters
    r = params[0]
    B = params[1] # total cap
    sec_params = params[2]

    # store number of sectors
    N_secs = np.shape(sec_params)[0]

    # find implied carbon price from Ts and compare to control carbon price
    tmp_price = cprice(Ts, B, r, sec_params)
    price_diff = tmp_price - price

    # loop through sectors and find the abatement at time T, compare to emissions rate
    a_iTs = []
    for i in range(N_secs):
        tmp_abar = sec_params[i][1] # tmp abar
        tmp_a = aT(Ts[i], tmp_price, r, sec_params[i])
        a_iTs.append(tmp_a[0] - tmp_abar)
    
    # make final list and return
    final_list = a_iTs
    final_list.append(price_diff[0])
    return final_list

def get_delay(Bp, params):
    """Get the premium amount of carbon emissions required for some amount of
    delay in decarbonization.

    The model allows us to tune the premium amount of emissions allocated to the
    politically challenged sectors, not the number of years we delay decarbonizing in that sector.
    This function computes the B_p that leads to a delay_amount years delay in decarbonizing.

    This function is meant to be an input to scipy.optimize.fsolve.

    Parameters
    ----------
    Bp: float
        the premium allocation of emissions to the politically challenged sector

    params: list
        list of required parameters
        - params[0]: delay amount, float, number of years to delay decarbonization
        - params[1]: chal_sec_ind, int, the challenged sector's numerical index in the sectoral parameters array
        - params[2]: r, float, social discount rate
        - params[3]: sec_params, (N_secs, 4) list, sectoral parameters, including optimal emissions allocation in perfect policy
    """

    # parse parameters
    delay_amount = params[0]
    chal_sec_ind = params[1]
    r = params[2]
    sec_params = params[3]

    # pull challenged sector's parameters and emissions allocation
    chal_sec_params = sec_params[chal_sec_ind]
    Bstar = chal_sec_params[-1]

    # note challenged sectors parameters without the emissions allocation
    chal_sec_params_noBstar = [chal_sec_params[:-1]]

    # get decarbonization time for perfect model with Bstar cap
    no_const = fsolve(get_decarb_time_and_price, x0=(40, 40),
                      args=[r, Bstar, chal_sec_params_noBstar])
    T_no_const = no_const[0]

    # get decarbonization with imperfect model with Bstar + Bp cap
    w_const = fsolve(get_decarb_time_and_price, x0=(40, 40),
                     args=[r, Bstar + Bp, chal_sec_params_noBstar])
    T_w_const = w_const[0]

    # return (T_w_const = T_no_const + delay_amount) in root-finder form
    return T_w_const - T_no_const - delay_amount

def cost_pre(t, args):
    """Compute total cost of policy before decarbonization date at time t.

    Note this function is meant to be an input to scipy.integrate.quad

    Parameters
    ----------
    t: float
        time

    args: (4,) list
        list of necessary parameters
        - args[0]: T, float, decarbonization time
        - args[1]: price, float, the carbon price
        - args[2]: r, float, social discount rate
        - params[3]: (3,) list, sector parameters
            - params[0]: delta, float, captial depreciation rate
            - params[1]: alpha, float, emissions rate in sector
            - params[2]: c, float, marginal investment cost

    Returns
    -------
    cost: float
        the cost at time t before decarbonization date T
    """

    # parse arguments and parameters
    T, price, r, params = args
    delta, alpha, c, t0 = params

    # compute pieces of optimal investment path - see Mathematica or paper for
    # equation
    ## first piece
    first_piece = c * alpha * delta * np.exp((r + delta) * -1 * T)

    ## second piece
    second_piece = np.exp(-t * delta) - np.exp(-T * delta)
    second_piece *= price / delta
    second_piece *= np.exp(-r * t0)

    # combine first and second piece of optimal investment,
    # plug into cost function with discounting piece in front
    total = first_piece + second_piece
    total *= np.exp(t * (r + delta)) / c

    cost = np.exp(-r * t) * 0.5 * c * total**2

    # return cost at time t
    return cost

def cost_post(t, args):
    """Compute total cost of policy after decarbonization date at time t.

    Note this function is meant to be an input to scipy.integrate.quad

    Parameters
    ----------
    t: float
        time

    args: (4,) list
        list of necessary parameters
        - args[0]: T, float, decarbonization time
        - args[1]: price, float, the carbon price
        - args[2]: r, float, social discount rate
        - params[3]: (3,) list, sector parameters
            - params[0]: delta, float, captial depreciation rate
            - params[1]: alpha, float, emissions rate in sector
            - params[2]: c, float, marginal investment cost

    Returns
    -------
    cost: float
        the cost at time t before decarbonization date T
    """

    # unpack parameters and arguments to function
    T, price, r, params = args
    delta, alpha, c, t0 = params 

    # after T, we're in the steadys state, so x = alpha * delta, and the
    # costs are only time dependent because of the discounting piece
    steady_state = np.exp(-r * t) * c * (alpha * delta)**2 * 0.5

    # return steady state cost
    return steady_state


def total_cost_onesec(args):
    """Compute total cost for one sector.

    Integrate the pre- and post-decarbonization cost for one sector.

    Parameters
    ----------
    args: (4,) list
        list of necessary parameters
        - args[0]: T, float, decarbonization time
        - args[1]: price, float, the carbon price
        - args[2]: r, float, social discount rate
        - params[3]: (3,) list, sector parameters
            - params[0]: delta, float, captial depreciation rate
            - params[1]: alpha, float, emissions rate in sector
            - params[2]: c, float, marginal investment cost
    
    Returns
    -------
    total_cost: float
        the total cost from 0 to infinity
    """

    # unpack parameters and arguments
    T, price, r, params = args
    delta, alpha, c, t0 = params

    # integrate using scipy.integrate.quad the pre- and post-decarbonization
    # costs
    pre_T_piece = quad(cost_pre, t0, T, args=args)[0]
    post_T_piece = quad(cost_post, T, np.inf, args=args)[0]

    # compute total cost and return
    total_cost = pre_T_piece + post_T_piece
    return total_cost


def total_cost(Ts, prices, params, r, chal_ind):
    """Compute total cost across a set of sectors.

    Parameters
    ----------
    Ts: (N_secs,) list
        list of decarbonization dates for sectors

    prices: (N_secs,) list
        carbon prices for each sector

    params: (N_secs, 4) list
        list of sector specific parameters, including optimal emissions
        allocation

    r: float
        social discount rate

    chal_ind: int
        challeneged sector's index in params, Ts, and prices list

    Returns
    -------
    costs: (N_secs,) list
        list of cost in each sector, with costs[chal_ind] being the challenged
        sector's total cost
    """

    # make empty array of costs
    costs = np.zeros_like(Ts, dtype=float)

    # define challenged sector's parameters, decarbonization date,
    # and carbon price
    T_chal = Ts[chal_ind]
    price_chal = prices[chal_ind]
    params_chal = params[chal_ind]

    # define other sectors' parameters, decarbonization dates,
    # and carbon prices
    Ts_exp = np.delete(Ts, chal_ind)
    prices_exp = np.delete(prices, chal_ind)
    params_exp = np.delete(params, chal_ind, axis=0)

    # compute cost of challenged sector
    chal_cost = total_cost_onesec([T_chal, price_chal, r, params_chal])
    costs[chal_ind] = chal_cost

    # loop through remaining sectors and compute the cost of those sectors
    for i in range(len(prices_exp)):
        tmp_exp_cost = total_cost_onesec([Ts_exp[i], prices_exp[i], r,
                                          params_exp[i]])

        # if we're below the challenged index, just add to list, if we're
        # above, make space by one entry
        if i < chal_ind:
            costs[i] = tmp_exp_cost
        else:
            costs[i+1] = tmp_exp_cost

    # return list
    return costs
