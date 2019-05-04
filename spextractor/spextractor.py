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

from sklearn.cluster import DBSCAN, MeanShift

import GPy

plt.ioff()

# Element, rest wavelength, low_1, high_1, low_2, high_2
LINES_Ia = [('Ca II H&K', 3945.12, 3450, 3800, 3800, 3950),
            ('Si 4000A', 4129.73, 3840, 3950, 4000, 4200),
            ('Mg II 4300A', 4481.2, 4000, 4250, 4300, 4700),
            ('Fe II 4800A', 5083.42, 4300, 4700, 4950, 5600),
            ('S W', 5536.24, 5050, 5300, 5500, 5750),
            ('Si II 5800A', 6007.7, 5400, 5700, 5800, 6000),
            ('Si II 6150A', 6355.1, 5800, 6100, 6200, 6600)
            ]

LINES_Ib = [('Fe II', 5169, 4950, 5050, 5150, 5250),
            ('He I', 5875, 5350, 5450, 5850, 6000)]

LINES_Ic = [('Fe II', 5169, 4950, 5050, 5150, 5250),
            ('O I', 7773, 7250, 7350, 7750, 7950)]

LINES = dict(Ia=LINES_Ia, Ib=LINES_Ib, Ic=LINES_Ic)


def pEW(wavelength, flux, cont_coords):
    '''
    Calculates the pEW between two chosen points.(cont_coords to be
    inputed as np.array([x1,x2], [y1,y2])
    '''
    cont = interpolate.interp1d(cont_coords[0], cont_coords[1],
                                bounds_error=False,
                                fill_value=1)  # define pseudo continuum with cont_coords
    nflux = flux / cont(
        wavelength)  # normalize flux within the pseudo continuum
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
        print(prev.message, e.message, filename)
        raise e


def _get_speed(lambda_m, lambda_m_err, lambda_rest):
    # This is the absolute minimum, save it.
    c = 299.792458
    l_quot = lambda_m / lambda_rest
    velocity = -c * (l_quot ** 2 - 1) / (l_quot ** 2 + 1)
    velocity_err = c * 4 * l_quot / (l_quot ** 2 + 1) ** 2 * lambda_m_err / lambda_rest
    return velocity, velocity_err


def compute_speed(lambda_0, x_values, y_values, m, plot):
    # Just pick the strongest
    min_pos = y_values.argmin()
    if min_pos == 0 or min_pos == y_values.shape[0]:
        # Feature not found
        return np.nan, np.nan

    lambda_m = x_values[min_pos]

    # To estimate the error, we sample possible spectra from the posterior and find the minima.
    samples = m.posterior_samples_f(x_values[:, np.newaxis], 100).squeeze().argmin(axis=0)
    lambda_m_samples = x_values[samples]
    lambda_m_err = lambda_m_samples.std()

    velocity, velocity_err = _get_speed(lambda_m, lambda_m_err, lambda_0)

    if plot:
        plt.axvline(lambda_m, color='k', linestyle='--')

    return velocity, velocity_err


def compute_speed_high_velocity(lambda_0, x_values, y_values, m, plot, method='MeanShift'):
    min_pos = y_values.argmin()
    if min_pos == 0 or min_pos == y_values.shape[0]:
        # Feature not found
        return [], [], np.nan, np.nan, [], []

    # Find all relative minima
    minima = signal.argrelmin(y_values, order=10)[0].tolist()
    # Append the global minimum (it could be near the edges)
    abs_min = y_values.argmin()
    minima.append(abs_min)
    # Remove redundancies and sort.
    minima = sorted(set(minima))

    # To estimate the error, we sample possible spectra from the posterior and find the minima.
    samples = m.posterior_samples_f(x_values[:, np.newaxis], 100).squeeze()

    minima_samples = []
    for i in range(samples.shape[1]):
        positions = signal.argrelmin(samples[:, i], order=10)[0]
        minima_samples.extend(positions)

    minima_samples = np.array(minima_samples)[:, np.newaxis]
    if method.lower() == 'dbscan':
        labels = DBSCAN(eps=1).fit_predict(minima_samples)
    elif method.lower() == 'meanshift':
        labels = MeanShift(10).fit_predict(minima_samples)
    else:
        raise ValueError('Invalid method {}, valid are MeanShift and DBSCAN'.format(method))

    velocity, velocity_err = compute_speed(lambda_0, x_values, y_values, m, plot=False)
    lambdas = []
    lambdas_err = []
    vel_hv = []
    vel_hv_err = []

    for x in np.unique(labels):
        if x == -1:
            # This is the "noise" label in DBSCAN
            continue

        matching = labels == x
        if matching.sum() < 5: continue  # This is just noise

        min_pos = minima_samples[matching]
        lambda_m = np.mean(x_values[min_pos])
        lambda_m_err = np.std(x_values[min_pos])

        lambdas.append(lambda_m)
        lambdas_err.append(lambda_m_err)

        this_v, this_v_err = _get_speed(lambda_m, lambda_m_err, lambda_0)
        vel_hv.append(this_v)
        vel_hv_err.append(this_v_err)

        if plot:
            plt.vlines(lambda_m, y_values[min_pos] - 0.2,
                       y_values[min_pos] + 0.2, color='k', linestyle='--')
    return lambdas, lambdas_err, velocity, velocity_err, vel_hv, vel_hv_err


def _filter_outliers(wavel, flux, sigma_outliers):
    """
    Attempt to remove sharp lines (teluric, cosmic rays...).

    First applies a heavy downsampling and then discards points that are
    further than sigma_outliers standard deviations
    """

    downsampling = 20
    x = wavel[::downsampling, np.newaxis]
    y = flux[::downsampling, np.newaxis]
    kernel = GPy.kern.Matern32(input_dim=1, lengthscale=300, variance=0.001)
    m = GPy.models.GPRegression(x, y, kernel)
    m.optimize()

    pred, var_ = m.predict(wavel[:, np.newaxis])
    sigma = np.sqrt(var_.squeeze())
    valid = np.abs(flux - pred.squeeze()) < sigma_outliers * sigma

    wavel = wavel[valid]
    flux = flux[valid]

    print('Auto-removed {} data points'.format(len(valid) - valid.sum()))

    return wavel, flux


def process_spectra(filename, z, downsampling=None, plot=False, type='Ia',
                    sigma_outliers=None, high_velocity=False, auto_prune=True,
                    remove_gaps=True, hv_clustering_method='MeanShift'):
    t00 = time.time()
    wavel, flux = load_spectra(filename, z)

    if remove_gaps:
        # Mask out regions where flux is 0
        keep = flux != 0
        wavel = wavel[keep]
        flux = flux[keep]

    if isinstance(type, str):
        lines = LINES[type]
    else:
        lines = type

    if auto_prune:
        # Restrict to the possible range of the lines, with some leeway
        wav_min = min(min(x for x in l if isinstance(x, (int, float))) for l in lines) - 500
        wav_max = max(max(x for x in l if isinstance(x, (int, float))) for l in lines) + 500
        i0, i1 = np.searchsorted(wavel, [wav_min, wav_max])

        wavel = wavel[i0:i1]
        flux = flux[i0:i1]
        flux /= flux.max()  # If this raises an error, the spectrum is empty

    if sigma_outliers is not None:
        # Remove spikes
        wavel, flux = _filter_outliers(wavel, flux, sigma_outliers)
        # And re-normalise
        flux /= flux.max()

    if downsampling is not None:
        wavel = wavel[::downsampling]
        flux = flux[::downsampling]

    x = wavel[:, np.newaxis]
    y = flux[:, np.newaxis]

    if plot:
        plt.figure()
        plt.title(filename)
        plt.xlabel(r"$\mathrm{Rest\ wavelength}\ (\AA)$", size=14)
        plt.ylabel(r"$\mathrm{Normalised\ flux}$", size=14)
        plt.plot(wavel, flux, color='k', alpha=0.5)

    kernel = GPy.kern.Matern32(input_dim=1, lengthscale=300, variance=0.001)
    # kernel = GPy.kern.RBF(input_dim=1, lengthscale=300, variance=0.01)
    m = GPy.models.GPRegression(x, y, kernel)
    m['Gaussian.noise.variance'][0] = 0.0027

    print('Created GP')
    t0 = time.time()
    m.optimize()
    print('Optimised in', time.time() - t0, 's.')
    print(m)

    mean, variance = m.predict(x)
    conf = np.sqrt(variance)
    if plot:
        print('Plotting')
        plt.plot(x, mean, color='red')
        plt.fill_between(x[:, 0], mean[:, 0] - conf[:, 0],
                         mean[:, 0] + conf[:, 0],
                         alpha=0.3, color='red')

    pew_results = dict()
    pew_err_results = dict()
    velocity_results = dict()
    veolcity_err_results = dict()

    lambda_hv_results = dict()
    lambda_hv_err_results = dict()

    vel_hv_results = dict()
    vel_hv_err_results = dict()

    t0_pew = time.time()
    for line_data in lines:
        element, rest_wavelength, low_1, high_1, low_2, high_2 = line_data

        # For n:th feature:
        cp_1 = np.searchsorted(x[:, 0], (low_1, high_1))
        index_low, index_hi = cp_1
        cp_2 = np.searchsorted(x[:, 0], (low_2, high_2))
        index_low_2, index_hi_2 = cp_2

        if index_low == index_hi or index_low_2 == index_hi_2:
            # Feature outside of range of the spectrum
            if high_velocity:
                lambda_hv_results[element] = []
                lambda_hv_err_results[element] = []
                vel_hv_results[element] = []
                vel_hv_err_results[element] = []
            velocity_results[element] = np.nan
            veolcity_err_results[element] = np.nan
            pew_results[element] = np.nan
            pew_err_results[element] = np.nan
            continue

        max_point = index_low + np.argmax(mean[index_low: index_hi])
        max_point_2 = index_low_2 + np.argmax(mean[index_low_2: index_hi_2])

        # Get the coordinates of the points:
        cp1_x, cp1_y = x[max_point, 0], mean[max_point, 0]
        cp2_x, cp2_y = x[max_point_2, 0], mean[max_point_2, 0]

        # Speed calculation -------------------
        if high_velocity:
            line_out = compute_speed_high_velocity(rest_wavelength, x[max_point:max_point_2, 0],
                                                   mean[max_point:max_point_2, 0], m, plot, hv_clustering_method)

            lambda_hv, lambda_hv_err, vel, vel_errors, vel_hv, vel_hv_err = line_out
            lambda_hv_results[element] = lambda_hv
            lambda_hv_err_results[element] = lambda_hv_err
            vel_hv_results[element] = vel_hv
            vel_hv_err_results[element] = vel_hv_err
        else:
            vel, vel_errors = compute_speed(rest_wavelength, x[max_point:max_point_2, 0],
                                            y[max_point:max_point_2, 0], m,
                                            plot)
        velocity_results[element] = vel
        veolcity_err_results[element] = vel_errors

        if np.isnan(vel):
            # The feature was not detected, set the PeW to NaN.
            pew_results[element] = np.nan
            pew_err_results[element] = np.nan
            continue

        # PeW calculation ---------------------
        pew_computed, pew_err = pEW(wavel, flux,
                                    np.array([[cp1_x, cp2_x], [cp1_y, cp2_y]]))
        pew_results[element] = pew_computed
        pew_err_results[element] = pew_err

        # Plotting the pew regions ------------
        if plot:
            plt.scatter([cp1_x, cp2_x], [cp1_y, cp2_y], color='k', s=80)
            _x_pew = np.linspace(cp1_x, cp2_x)
            _m_pew = (cp2_y - cp1_y) / (cp2_x - cp1_x)
            _y_pew_hi = _m_pew * _x_pew + cp1_y - _m_pew * cp1_x
            _y_pew_low = m.predict(_x_pew[:, None])[0][:, 0]
            plt.fill_between(_x_pew, _y_pew_low, _y_pew_hi, color='y',
                             alpha=0.3)

    print('pEWs computed in {:.2f} s.'.format(time.time() - t0_pew))
    print(time.time() - t00, 's')
    if high_velocity:
        outputs = pew_results, pew_err_results, velocity_results, veolcity_err_results, lambda_hv_results, lambda_hv_err_results, vel_hv_results, vel_hv_err_results, m
    else:
        outputs = pew_results, pew_err_results, velocity_results, veolcity_err_results, m
    return outputs
