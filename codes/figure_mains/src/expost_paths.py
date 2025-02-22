"""Generate paths for state variables after simulations of decarbonization
dates and carbon prices have been done.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
5.28.2024

To run: expost_paths.py [cal] [save_output] 
"""

import numpy as np


def get_investment_path(T, delta, abar, r, mu, cbar, t0=0, Tmax=150,
                        dt=0.0001):
    """Return the investment path.
    """

    times = np.arange(0, Tmax + dt, dt)
    x_it = np.zeros_like(times, dtype=float)

    # before decarbonization, use the path form
    # first piece is just steady state bit
    times_pre = times[times <= T]
    first_piece = delta * abar * np.exp((r + delta) * (times_pre - T))

    # second piece
    second_piece = 1 - np.exp(delta * (times_pre - T))
    second_piece *= mu * (delta * cbar)**(-1) * np.exp(r * times_pre)
    second_piece *= np.exp(-r * t0)

    # add for actual path
    x_it[times <= T] = first_piece + second_piece

    # after decarbonization, we're in the steady state
    x_it[times > T] = delta * abar

    # set before t0 to zero
    x_it[times < t0] = 0

    return x_it


def get_abatement_path(T, delta, abar, r, mu, cbar, t0=0, Tmax=150, dt=0.0001):
    """Return abatement path.
    """

    times = np.arange(0, Tmax + dt, dt)
    ait = np.zeros_like(times, dtype=float)

    # before decarbonization, use path form
    # first piece is steady state bit
    times_pre = times[times <= T]
    first_piece = np.exp(t0 * delta)\
        - np.exp(r * (times_pre - t0) + times_pre * delta)
    first_piece *= -1 * mu / (cbar * (r + delta))

    second_piece = np.exp(t0 * (r + 2 * delta))\
        - np.exp(times_pre * (r + 2 * delta))
    second_piece *= mu / (cbar * (r + 2 * delta))
    second_piece *= np.exp(-r * t0 - T * delta)

    third_piece = np.exp(times_pre * (r + 2 * delta))\
        - np.exp(t0 * (r + 2 * delta))
    third_piece *= abar * delta**2 / (r + 2 * delta)
    third_piece *= np.exp(-T * (r + delta))

    total = first_piece + second_piece + third_piece
    total *= np.exp(-times_pre * delta) / delta

    ait[times <= T] = total
    ait[times > T] = abar
    ait[times < t0] = 0

    # WILL COMPLETE IF NEEDED
    return ait
