# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:13:56 2015
Automated pEW and velocity extractor.
@author: Seméli Papadogiannakis
"""
from __future__ import division, print_function
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

from scipy import interpolate
from scipy import signal

import GPy

plt.ioff()

LINES = dict(((1, 3945.12), (2, 4129.73), (3, 4481.2), (4, 5083.42),
              (5, 5536.24), (6, 6007.7), (7, 6355.1)))


def pEW(wavelength, flux, cont_coords):
    '''
    Calculates the pEW between two chosen points.(cont_coords to be
    inputed as np.array([x1,x2], [y1,y2])
    '''
    cont = interpolate.interp1d(cont_coords[0], cont_coords[1], bounds_error=False,
                                fill_value=1)  # define pseudo continuum with cont_coords
    nflux = flux / cont(wavelength)  # normalize flux within the pseudo continuum
    pEW = 0
    for i in range(len(wavelength)):
        if wavelength[i] > cont_coords[0, 0] and wavelength[i] < cont_coords[0, 1]:
            dwave = 0.5 * (wavelength[i + 1] - wavelength[i - 1])
            pEW += dwave * (1 - nflux[i])

    flux_err = np.abs(signal.cwt(flux, signal.ricker, [1])).mean()
    pEW_stat_err = flux_err
    pEW_cont_err = np.abs(cont_coords[0, 0] - cont_coords[0, 1]) * flux_err
    pEW_err = math.hypot(pEW_stat_err, pEW_cont_err)
    return pEW, pEW_err


def gaussian(x, AB, mean, sigma, baseline):
    '''
   x = in this case normalised time
   AB = Amplitude of the secondary bump
   x0 = The normalised time of the secondary bump max
   sigma = width of the gaussian
   '''
    return baseline + AB * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


# Import the spectra
def load_spectra(filename, z):
    '''
    Loading spectra of different types, except .fits files.
    '''
    # Method 1
    try:
        data = ascii.read(filename)
        flux = data[:, 1]
        wavel = data[:, 0]
        flux /= flux.max()
        return wavel, flux
        '''
        # Make sure there is data around silicon:
        if wavel.min() > 5800 or wavel.max() < 7000:
            return None

        idx1, idx2 = np.searchsorted(wavel, [3000, 8000])

        # Load only the data between 3000 and 8000 A
        wavel = data[idx1:idx2]['col1'] / (1 + z)
        flux = data[idx1:idx2]['col2']
        flux /= flux.max()  # normalise intensity
        return wavel, flux
        '''
    except Exception as e:
        prev = e

    # Method 2
    try:
        data = np.genfromtxt(filename)
        wavel = data[:, 0]
        wavel = wavel / (1 + z)
        '''
        # Make sure there is data around silicon:
        if wavel.min() > 5800 or wavel.max() < 7000:
            return None
        idx1, idx2 = np.searchsorted(wavel, [3000, 8000])

        wavel = wavel[idx1:idx2]
        flux = data[idx1:idx2, 1]
        flux /= flux.max()  # normalise intensity
        '''
        flux = data[:, 1]
        flux /= flux.max()
        return wavel, flux
    except Exception as e:
        print(prev.message, e.message)


def compute_speed(feature_n, x_values, y_values):
    from scipy import signal
    for index in signal.argrelmin(y_values, order=10)[0]:
        lambda_m = x_values[index]
        # lambda_m = x_values[y_values.argmin()]
        c = 299.792458
        lambda_0 = LINES[feature_n]  # Restframe
        l_quot = lambda_m / lambda_0
        velocity = -c * (l_quot ** 2 - 1) / (l_quot ** 2 + 1)
        if __name__ == '__main__':
            print("f{} velocity: {:.3f}".format(feature_n, velocity))
            plt.axvline(lambda_m, color='b')
    return velocity


def process_spectra(filename, z, downsampling=None, plot=False):
    t00 = time.time()
    wavel, flux = load_spectra(filename, z)
    if downsampling is not None:
        wavel = wavel[::downsampling]
        flux = flux[::downsampling]

    x = wavel[:, np.newaxis]
    y = flux[:, np.newaxis]

    plt.figure()
    plt.title(filename)
    plt.xlabel(r"$\mathrm{Rest\ wavelength}\ (\AA)$", size=14)
    plt.ylabel(r"$\mathrm{Normalised\ flux}$", size=14)
    plt.plot(wavel, flux, color='k', alpha=0.5)

    kernel = GPy.kern.Matern32(input_dim=1, lengthscale=300, variance=0.1)
    m = GPy.models.GPRegression(x, y, kernel)
    print('Created GP')
    t0 = time.time()
    m.optimize()
    print('Optimised in', time.time() - t0, 's.')
    print(m)

    if plot:
        print('Plotting')
        mean, conf = m.predict(x)
        plt.plot(x, mean, color='red')
        plt.fill_between(x[:, 0], mean[:, 0] - conf[:, 0], mean[:, 0] + conf[:, 0],
                         alpha=0.3, color='red')

    pew_results = dict()
    pew_err_results = dict()
    velocity_results = dict()

    t0_pew = time.time()
    for n in range(1, 8):
        if n == 1:  # Ca II H&K
            # low_1 = 3450
            low_1 = 3580
            high_1 = 3800
            low_2 = 3800
            # high_2 = 4100
            high_2 = 3950
        elif n == 2:  # Si 4000A
            # low_1 = 3900
            low_1 = 3840
            high_1 = 3950
            low_2 = 4000
            high_2 = 4200
        elif n == 3:  # Mg II 4300A
            low_1 = 4000
            high_1 = 4250
            low_2 = 4300
            high_2 = 4700
        elif n == 4:  # Fe II 4800A
            low_1 = 4300
            high_1 = 4700
            low_2 = 4950
            high_2 = 5600
        elif n == 5:  # Si W
            low_1 = 5050
            high_1 = 5300
            low_2 = 5500
            high_2 = 5750
        elif n == 6:  # Si II 5800A
            low_1 = 5400
            high_1 = 5700
            low_2 = 5800
            high_2 = 6000
        elif n == 7:  # Si II 6150A
            low_1 = 5800
            high_1 = 6100
            low_2 = 6200
            high_2 = 6600

        # For n:th feature:
        cp_1 = np.searchsorted(x[:, 0], (low_1, high_1))
        index_low, index_hi = cp_1
        max_point = index_low + np.argmax(mean[index_low: index_hi])

        cp_2 = np.searchsorted(x[:, 0], (low_2, high_2))
        index_low_2, index_hi_2 = cp_2
        max_point_2 = index_low_2 + np.argmax(mean[index_low_2: index_hi_2])

        # Get the coordinates of the points:
        cp1_x, cp1_y = x[max_point, 0], mean[max_point, 0]
        cp2_x, cp2_y = x[max_point_2, 0], mean[max_point_2, 0]

        # PeW calculation ---------------------
        pew_computed, pew_err = pEW(wavel, flux,
                                    np.array([[cp1_x, cp2_x], [cp1_y, cp2_y]]))

        # Plotting the pew regions ------------
        if plot:
            plt.scatter([cp1_x, cp2_x], [cp1_y, cp2_y], color='k', s=80)
            _x_pew = np.linspace(cp1_x, cp2_x)
            _m_pew = (cp2_y - cp1_y) / (cp2_x - cp1_x)
            _y_pew_hi = _m_pew * _x_pew + cp1_y - _m_pew * cp1_x
            _y_pew_low = m.predict(_x_pew[:, None])[0][:, 0]
            plt.fill_between(_x_pew, _y_pew_low, _y_pew_hi, color='y', alpha=0.3)

        # Velocity calculation ----------------
        if n in LINES:
            # compute_speed_fit(n, wavel, flux)
            vel = compute_speed(n, x[max_point:max_point_2, 0],
                                mean[max_point:max_point_2, 0])
            velocity_results[n] = vel
        # print 'pEW {:.2f} +- {:.2f}'.format(pew_computed, pew_err)

        pew_results[n] = pew_computed
        pew_err_results[n] = pew_err

    print('pEWs computed in {:.2f} s.'.format(time.time() - t0_pew))
    print(time.time() - t00, 's')
    return pew_results, pew_err_results, velocity_results
