from util import c_inv_paper_series

import scipy.stats
import os
import re
import sys
import datetime
from datetime import date
from socket import getfqdn
from scipy.interpolate import interp2d
import math
import pkg_resources # installed by setuptools package
from distutils.version import LooseVersion
from scipy.io import loadmat
import scipy
import scipy.stats as stts
from types import new_class
import numpy as np
from numpy import pi, isnan, nanmin, nanmax
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datajoint as dj
import seaborn as sns
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.signal import chirp, find_peaks, peak_widths 
from djd.revcorr import RevCorr, RCVar, RCDtOpt, RCParams, RCResponse, RCSVD, RCSpatialAuto
from djd.stimulus import Stimulus
from djd.unit import Unit, WaveShapeCluster
from djd.tuning import RCTun, RCOSI
from djd.series import Series
from djd.collab import ContrastInvariance
from djd.mouse import Mouse
from djd.util import get_t_kernels, get_P_spike_given_stim, binarize_events, count_stim_spikes, one_event_in_interval, event_in_interval, count_stim_bins
from scipy.interpolate import interp1d
from adjustText import adjust_text
import statistics
from scipy.optimize import curve_fit
from lmfit import Parameters, minimize, report_fit, fit_report, conf_interval, printfuncs, Minimizer
import seaborn
from mycolorpy import colorlist as mcp
from termcolor import colored

np.seterr(divide='ignore', invalid='ignore')


def von_mises(x, pref_ori=90, rp=1, r0=0, kappa=1, period=180):
    """
    A von Mises function living on a circle of adjustable period.

    The von Mises function is a smooth variant of the circular Gaussian. This
    function implements an amplitude-normalized version of the von Mises
    distribution, the purpose of which is to uncouple amplitude and width.
    It also transforms the width parameter kappa by squaring to achieve more linear
    scaling of the width.

    Parameters
    ----------cal_average(num)
    x : array-like
        orientations in degrees
    pref_ori : float between 0 and period
        preferred orientation
    rp : float
        response to the preferred orientation
    r0 : float
        background response
    kappa : float
        width parameter
    period : float
        circular period

    Returns
    -------
    y : array-like
        response at orientation x
    """
    omega = 360/period
    oris = x * np.pi/180
    po = pref_ori * np.pi/180
    amp = rp-r0
    return r0 + amp*np.exp(kappa**2*(np.cos(omega*(oris-po))-1))

def von_mises_modulated(x, A=1, pref_ori=90, rp=1, r0=0, kappa=1, period=180, C = 1, b0=0):
    """
    A von Mises function living on a circle of adjustable period.

    The von Mises function is a smooth variant of the circular Gaussian. This
    function implements an amplitude-normalized version of the von Mises
    distribution, the purpose of which is to uncouple amplitude and width.
    It also transforms the width parameter kappa by squaring to achieve more linear
    scaling of the width.

    Parameters
    ----------
    x : array-like
        orientations in degrees
    pref_ori : float between 0 and period
        preferred orientation
    rp : float
        response to the preferred orientation
    r0 : float
        background response
    kappa : float
        width parameter
    period : float
        circular period

    Returns
    -------
    y : array-like
        response at orientation x
    """
    omega = 360/period
    oris = x * np.pi/180
    po = pref_ori * np.pi/180
    amp = rp-r0
    y = C*r0 + A*amp*np.exp(kappa**2*(np.cos(omega*(oris-po))-1)) + b0
    # y = A*(r0 + amp*np.exp(kappa**2*(np.cos(omega*(oris-po))-1)))
    # y = amp*np.exp(kappa**2*(np.cos(omega*(oris-po))-1))
    return y


def von_mises_new(x, pref_ori=90, rp=1, r0=0, kappa=1, period=180, b0 = 0):
    """
    A von Mises function living on a circle of adjustable period.

    The von Mises function is a smooth variant of the circular Gaussian. This
    function implements an amplitude-normalized version of the von Mises
    distribution, the purpose of which is to uncouple amplitude and width.
    It also transforms the width parameter kappa by squaring to achieve more linear
    scaling of the width.

    Parameters
    ----------cal_average(num)
    x : array-like
        orientations in degrees
    pref_ori : float between 0 and period
        preferred orientation
    r0 : float
        background response
    kappa : float
        width parameter
    period : float
        circular period

    Returns
    -------
    y : array-like
        response at orientation x
    """
    omega = 360/period
    oris = x * np.pi/180
    po = pref_ori * np.pi/180
    return r0 + (rp-r0)*np.exp(kappa*(np.cos(omega*(oris-po))-1)) + b0


def wrapped_gaussian(x, pref_ori=90, rp=1, r0=0, sigma=30, period=180, n=None):
    """
    A smooth orientation tuning function constructed from gaussians shifted by the period.

    By using an infinite sum of gaussians with preferred orientations at phi, phi+period,
    phi+2*period, ... , one can create a smooth orientation tuning function. In practice a few
    gaussians are enough for most cases but if tuning widths become very large, a larger number
    of gaussians is needed to adequately represent the infinite sum. Taken from Swindale, 1998.

    Parameters
    ----------
    x : array-like
        orientations in degrees
    pref_ori : float between 0 and period
        preferred orientation
    rp : float
        response to the preferred orientation
    r0 : float
        background response
    sigma : float
        width parameter
    period : float
        circular period
    n : int
        number of gaussians in sum

    References
    ----------
    Swindale, N. V. (1998). Orientation tuning curves: Empirical description and estimation of
    parameters. Biological Cybernetics, 78(1), 45–56. https://doi.org/10.1007/s004220050411
    """
    def _wg(k):
        res = np.exp(-(x - pref_ori) ** 2 / (2 * sigma ** 2))
        for iii in range(k):
            res += np.exp(-(x - pref_ori + period * (iii + 1)) ** 2 / (2 * sigma ** 2))
            res += np.exp(-(x - pref_ori - period * (iii + 1)) ** 2 / (2 * sigma ** 2))
        try:
            res = res/np.max(res)
        except FloatingPointError:
            pass
        return (rp-r0) * res + r0

    if n is None:
        n = 0
        while not np.allclose(_wg(n), _wg(n+1)) and n < 20:
            n += 1
    else:
        pass
    return _wg(n)


def wrapped_gaussian_modulated(x, pref_ori=90, rp=1, r0=0, sigma=30, period=180, b0=0, n=None):
    """
    A smooth orientation tuning function constructed from gaussians shifted by the period.

    By using an infinite sum of gaussians with preferred orientations at phi, phi+period,
    phi+2*period, ... , one can create a smooth orientation tuning function. In practice a few
    gaussians are enough for most cases but if tuning widths become very large, a larger number
    of gaussians is needed to adequately represent the infinite sum. Taken from Swindale, 1998.

    Parameters
    ----------
    x : array-like
        orientations in degrees
    pref_ori : float between 0 and period
        preferred orientation
    rp : float
        response to the preferred orientation
    r0 : float
        background response
    sigma : float
        width parameter
    period : float
        circular period
    n : int
        number of gaussians in sum

    References
    ----------
    Swindale, N. V. (1998). Orientation tuning curves: Empirical description and estimation of
    parameters. Biological Cybernetics, 78(1), 45–56. https://doi.org/10.1007/s004220050411
    """
    def _wg(k):
        res = np.exp(-(x - pref_ori) ** 2 / (2 * sigma ** 2))
        for iii in range(int(k)):
            res += np.exp(-(x - pref_ori + period * (iii + 1)) ** 2 / (2 * sigma ** 2))
            res += np.exp(-(x - pref_ori - period * (iii + 1)) ** 2 / (2 * sigma ** 2))
        try:
            res = res/np.max(res)
        except FloatingPointError:
            pass
        return (rp-r0) * res + r0 + b0

    if n is None:
        n = 0
        while not np.allclose(_wg(n), _wg(n+1)) and n < 20:
            n += 1
    else:
        pass
    y = _wg(n)
    return y

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t  
    avg = sum_num / len(num)
    return avg  

#FUNCTIONS for fitting tuning curves with LMFIT
def von_mises_modulated_dataset_shared(params, i, x):
    A = params['A_%i' % (i+1)].value
    pref_ori = params['pref_ori'].value
    rp = params['rp'].value
    r0 = params['r0'].value
    kappa = params['kappa_%i' % (i+1)].value
    period = params['period'].value
    C = params['C_%i' % (i+1)].value
    b0 = params['b0_%i' % (i+1)].value
    return von_mises_modulated(x,A,pref_ori,rp,r0,kappa,period,C,b0)

def von_mises_modulated_dataset_shared_A(params, i, x):
    A = params['A_%i' % (i+1)].value
    pref_ori = params['pref_ori'].value
    rp = params['rp'].value
    r0 = params['r0'].value
    kappa = params['kappa'].value
    period = params['period'].value
    C = params['C_%i' % (i+1)].value
    b0 = params['b0_%i' % (i+1)].value
    
    return von_mises_modulated(x,A,pref_ori,rp,r0,kappa,period,C,b0)

def von_mises_new_dataset_shared(params, i, x):
    pref_ori = params['pref_ori'].value
    rp = params['rp_%i' % (i+1)].value
    r0 = params['r0_%i' % (i+1)].value
    kappa = params['kappa_%i' % (i+1)].value
    period = params['period'].value
    b0 = params['b0_%i' % (i+1)].value
    return von_mises_new(x,pref_ori,rp,r0,kappa,period,b0)

def von_mises_new_dataset_shared_A(params, i, x):
    pref_ori = params['pref_ori'].value
    rp = params['rp_%i' % (i+1)].value
    r0 = params['r0_%i' % (i+1)].value
    kappa = params['kappa'].value
    period = params['period'].value
    b0 = params['b0_%i' % (i+1)].value
    return von_mises_new(x,pref_ori,rp,r0,kappa,period,b0)

def von_mises_new_ori_shared(params, i, x):
    pref_ori = params['pref_ori_%i' % (i+1)].value
    rp = params['rp_%i' % (i+1)].value
    r0 = params['r0_%i' % (i+1)].value
    kappa = params['kappa_%i' % (i+1)].value
    period = params['period'].value
    b0 = params['b0_%i' % (i+1)].value
    return von_mises_new(x,pref_ori,rp,r0,kappa,period,b0)

def von_mises_new_ori_shared_A(params, i, x):
    pref_ori = params['pref_ori_%i' % (i+1)].value
    rp = params['rp_%i' % (i+1)].value
    r0 = params['r0_%i' % (i+1)].value
    kappa = params['kappa'].value
    period = params['period'].value
    b0 = params['b0_%i' % (i+1)].value
    return von_mises_new(x,pref_ori,rp,r0,kappa,period,b0)


def objective_shared(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - von_mises_modulated_dataset_shared(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def objective_shared_A(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - von_mises_modulated_dataset_shared_A(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def objective_shared_new(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - von_mises_new_dataset_shared(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def objective_shared_new_A(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - von_mises_new_dataset_shared_A(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def objective_shared_ori_new(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - von_mises_new_ori_shared(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def objective_shared_ori_new_A(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - von_mises_new_ori_shared_A(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def wrapped_gaussian_modulated_dataset_shared(params, i, x):

    pref_ori = params['pref_ori'].value
    rp = params['rp_%i' % (i+1)].value
    r0 = params['r0_%i' % (i+1)].value
    sigma = params['sigma_%i' % (i+1)].value
    period = params['period'].value
    b0 = params['b0'].value
    # n = params['n_%i' % (i+1)].value
    return wrapped_gaussian_modulated(x,pref_ori,rp,r0,sigma,period,b0)

def wrapped_gaussian_modulated_dataset_shared_A(params, i, x):

    pref_ori = params['pref_ori'].value
    rp = params['rp_%i' % (i+1)].value
    r0 = params['r0'].value
    sigma = params['sigma'].value
    period = params['period'].value
    b0 = params['b0_%i' % (i+1)].value
    # n = params['n'].value
    return wrapped_gaussian_modulated(x,pref_ori,rp,r0,sigma,period,b0)

def objective_shared_gauss(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - wrapped_gaussian_modulated_dataset_shared(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def objective_shared_A_gauss(params, x, data):
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - wrapped_gaussian_modulated_dataset_shared_A(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()



def list_keys_new_e_i(keys):
    dfff_e_i = Unit().Spikes & (Stimulus() & keys) & keys
    list_keys_new_e_i = []
    for i in range(0, len(keys)):
        mouse = {'m':dfff_e_i.fetch('m')[i], 's': dfff_e_i.fetch('s')[i], 'e': dfff_e_i.fetch('e')[i], 'u': dfff_e_i.fetch('u')[i]}
        list_keys_new_e_i.append(mouse)
    return list_keys_new_e_i

def dtopt_indx_sorted(keys):
    dtrange = (RCParams & keys).fetch1('smoothdts')
    dtopt_df = pd.DataFrame(RCDtOpt & keys)
    dfff_e_i = Unit().Spikes & (Stimulus() & keys) & keys
    dfff_e_i_dt = pd.DataFrame(dfff_e_i).merge(dtopt_df, on=["m","s","e","u"])
    dtopt_sorted = dfff_e_i_dt.dt_maxresponse
    spk_times_all_units = dfff_e_i_dt.spk_times

    dtopt_indx_sorted = []
    for i in range(0, len(dtopt_sorted)):
        dtopt_indx_sorted.append(np.where(np.isclose(dtrange, dtopt_sorted[i]))[0][0])
    return dtopt_indx_sorted

def grat_cond_all(keys):
    grat_cond_all= []
    grat_cond_df_all= []
    keys_new = list_keys_new_e_i(keys)
    for i in range(0,len(keys_new)):
        grat_cond_all.append((Stimulus.GratingCond & keys_new[i]).fetch('stim_id', 'grat_orientation', 'grat_contrast'))
        df = pd.DataFrame({'stim_id':grat_cond_all[i][0], 'grat_orientation':grat_cond_all[i][1], 'grat_contrast':grat_cond_all[i][2]})
        df['stim_id'] = df['stim_id']
        df.set_index('stim_id', append=True, inplace=True)
        df = df.droplevel(level = 0)
        grat_cond_df_all.append(df)
    return grat_cond_all

def grat_cond_unit_j(j, keys):
    keys_new = list_keys_new_e_i(keys)
    grat_cond = (Stimulus.GratingCond & keys_new[j]).fetch('stim_id', 'grat_orientation', 'grat_contrast')
    df = pd.DataFrame({'stim_id':grat_cond[0], 'grat_orientation':grat_cond[1], 'grat_contrast':grat_cond[2]})
    df['stim_id'] = df['stim_id']
    df.set_index('stim_id', append=True, inplace=True)
    df = df.droplevel(level = 0)
    return grat_cond

def firing_rate_revcorr_df(j,keys):
    # 'm', 's', 'e', 'u', dt_maxresponse in dtopt_df
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    keys_new = list_keys_new_e_i(keys)
    dtopt_df = pd.DataFrame(RCDtOpt & keys)
    dfff_e_i = Unit().Spikes & (Stimulus() & keys) & keys
    dfff_e_i_dt = pd.DataFrame(dfff_e_i).merge(dtopt_df, on=["m","s","e","u"])
    dtopt_sorted = dfff_e_i_dt.dt_maxresponse
    spk_times_all_units = dfff_e_i_dt.spk_times

    trial = (Stimulus.Trial & keys_new[j]).df
    kernel = get_t_kernels(spk_times_all_units[j], trial, dtrange, precision, bin_spikes=True)
    firing_rates_df = kernel
    firing_rates_df = firing_rates_df.rename(columns={'p_spike_stim': 'firing_rates'})
    firing_rates_df['firing_rates'] = firing_rates_df['firing_rates'].apply(lambda x: x/(0.001))
    return firing_rates_df

def firing_rate_revcorr(j,keys):
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    keys_new = list_keys_new_e_i(keys)
    dtopt_indx = dtopt_indx_sorted(keys)
    firing_rates_df = firing_rate_revcorr_df(j, keys)
    delta_t = firing_rates_df.groupby('stim_id')['delta_t'].apply(list)
    fr = firing_rates_df.groupby('stim_id')['firing_rates'].apply(list)
    fr_list = []
    delta_t_list = []
    for i in range(0, firing_rates_df.stim_id.max() + 1):
        fr_list.append(fr[i])
        delta_t_list.append(delta_t[i])

    list_fr = []

    for i in range(0, firing_rates_df.stim_id.max() + 1):
        list_fr_short = list(zip(delta_t_list[i], fr_list[i]))
        list_fr_short = sorted(list_fr_short, key = lambda x: x[0])
        list_fr.append(list_fr_short)
    list_full_fr = []
    for k in range(0, len(dtrange)):
        list_full_fr_short = []
        for i in range(0, firing_rates_df.stim_id.max() + 1):
            list_full_fr_short.append(list_fr[i][k][1])
        list_full_fr.append(list_full_fr_short)
    return list_full_fr

def data_fr_max(keys, data_fr):
    data_fr_max = []
    data_fr_max_indx = []
    for j in range(0, len(keys)):
        short_list = []
        for i in range(0, 41):
            short_list.append(max(data_fr[j][i]))
        data_fr_max.append(max(short_list))
        data_fr_max_indx.append(short_list.index(max(short_list)))
    return data_fr_max, data_fr_max_indx

def table_firing_rate(j, keys, a, b, data_fr):
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    keys_new = list_keys_new_e_i(keys)
    # dtopt_indx = dtopt_indx_sorted(keys)
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    list_fr = firing_rate_revcorr(j,keys)
    table_x_y_unit = []
    for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
        # uncomment line to get unnormalized firing rate
        table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
        table_x_y = table_x_y.groupby('orient')['fr'].mean()
        list_x_y = list(table_x_y)
        list_x_y.append(list_x_y[0])
        table_x_y_unit.append(list_x_y)
    return table_x_y_unit

def table_firing_rate_contr(j, keys, a, b, data_fr):
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    keys_new = list_keys_new_e_i(keys)
    # dtopt_indx = dtopt_indx_sorted(keys)
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    list_fr = firing_rate_revcorr(j,keys)
    table_x_y_unit = []
    for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
        # uncomment line to get unnormalized firing rate
        table_x_y = pd.DataFrame({'contr': grat_cond_unit_j(j, keys)[2], 'fr': list_fr[w]})
        table_x_y = table_x_y.groupby('contr')['fr'].mean()
        list_x_y = list(table_x_y)
        list_x_y.append(list_x_y[0])
        table_x_y_unit.append(list_x_y)
    return table_x_y_unit

def table_firing_rate_all(j, keys, data_fr):
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    keys_new = list_keys_new_e_i(keys)
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    # dtopt_indx = dtopt_indx_sorted(keys)
    list_fr = firing_rate_revcorr(j,keys)
    table_x_y_unit = []
    for w in range(0, len(dtrange)):
        # uncomment line to get unnormalized firing rate
        table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
        table_x_y = table_x_y.groupby('orient')['fr'].mean()
        list_x_y = list(table_x_y)
        list_x_y.append(list_x_y[0])
        table_x_y_unit.append(list_x_y)
    return table_x_y_unit

def table_firing_rate_all_contr(j, keys, data_fr):
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    keys_new = list_keys_new_e_i(keys)
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    # dtopt_indx = dtopt_indx_sorted(keys)
    list_fr = firing_rate_revcorr(j,keys)
    table_x_y_unit = []
    for w in range(0, len(dtrange)):
        # uncomment line to get unnormalized firing rate
        table_x_y = pd.DataFrame({'contr': grat_cond_unit_j(j, keys)[2], 'fr': list_fr[w]})
        table_x_y = table_x_y.groupby('contr')['fr'].mean()
        list_x_y = list(table_x_y)
        list_x_y.append(list_x_y[0])
        table_x_y_unit.append(list_x_y)
    return table_x_y_unit

def firing_rate_loaded(j, keys, a, b, data_fr):
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    keys_new = list_keys_new_e_i(keys)
    # dtopt_indx = dtopt_indx_sorted(keys)
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    fr_a_b = data_fr[j][dtopt_indx[j]-a: dtopt_indx[j]+b]
    return fr_a_b

def isNaN(num):
    return num!= num    


def von_mises_multi_fit_shared_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr):
    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    pref_oris_all_units_topt = orient_unique[list(data_fr[j][dtopt_indx[j]]).index(max(list(data_fr[j][dtopt_indx[j]])))]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    data = np.array(table_x_y_expanded)
    fit_params = Parameters()
    for iy, y in enumerate(data):
        # fit_params.add('pref_ori', value=pref_oris_all_units_short[iy], min=0, max=180)
        fit_params.add('pref_ori', value=pref_oris_all_units_topt, min=0, max=180.5)
        fit_params.add('rp_%i' % (iy+1), value=max(np.array(table_x_y_expanded).flatten()), min=np.mean(np.array(table_x_y_expanded[iy])), max=max(np.array(table_x_y_expanded).flatten()) + np.mean(np.array(table_x_y_expanded[iy])))
        fit_params.add('r0_%i' % (iy+1), value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.mean(np.array(table_x_y_expanded[iy])))
        if tuning_width_all_units_topt >10:
            fit_params.add('kappa_%i' % (iy+1), value=0, min=0, max=1)
        elif isNaN(tuning_width_all_units_short[iy]) == True:
            fit_params.add('kappa_%i' % (iy+1), value=0, min=0, max=10)
        else:
            fit_params.add('kappa_%i' % (iy+1), value=np.sqrt(tuning_width_all_units_short[iy]), min=0.5, max=10)
        fit_params.add('period', value=180, min=80, max=185)
        fit_params.add('b0_%i' % (iy+1), value=0, min=-np.inf, max=np.inf)
    out = minimize(objective_shared_new, fit_params, args=(orient_unique, data))
    # to get quality of fit
    # mini = Minimizer(objective_shared_new, fit_params, fcn_args=(orient_unique, data))
    # ci = conf_interval(mini, out)
    # print(printfuncs.report_ci(ci))

    pref_ori_fit = []
    rp_fit = []
    r0_fit = []
    kappa_fit = []
    period_fit = []
    b0_fit = []
    
    for m in range(0,l):
        pref_ori_fit.append(out.params['pref_ori'].value)
        rp_fit.append(out.params['rp_%i' % (m+1)].value)
        r0_fit.append(out.params['r0_%i' % (m+1)].value)
        kappa_fit.append(out.params['kappa_%i' % (m+1)].value)
        period_fit.append(out.params['period'].value)
        # b0_fit.append(out.params['b0'].value)
        b0_fit.append(out.params['b0_%i' % (m+1)].value)
    # print(fit_report(out.params, show_correl=False)) # fit report
    return pref_ori_fit, rp_fit, r0_fit, kappa_fit, period_fit, b0_fit

def von_mises_multi_fit_shared_ori_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr):
    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = orient_unique[list(data_fr[j][dtopt_indx[j]]).index(max(list(data_fr[j][dtopt_indx[j]])))]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    data = np.array(table_x_y_expanded)
    fit_params = Parameters()
    for iy, y in enumerate(data):
        fit_params.add('pref_ori_%i' % (iy+1), value=pref_oris_all_units_topt, min=0, max=180)
        fit_params.add('rp_%i' % (iy+1), value=max(np.array(table_x_y_expanded).flatten()), min=np.mean(np.array(table_x_y_expanded[iy])), max=max(np.array(table_x_y_expanded).flatten()) + np.mean(np.array(table_x_y_expanded[iy])))
        fit_params.add('r0_%i' % (iy+1), value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.mean(np.array(table_x_y_expanded[iy])))
        if tuning_width_all_units_topt >10:
            fit_params.add('kappa_%i' % (iy+1), value=0, min=0, max=1)
        elif isNaN(tuning_width_all_units_short[iy]) == True:
            fit_params.add('kappa_%i' % (iy+1), value=0, min=0, max=10)
        else:
            fit_params.add('kappa_%i' % (iy+1), value=np.sqrt(tuning_width_all_units_short[iy]), min=0.5, max=10)
        fit_params.add('period', value=180, min=80, max=185)
        fit_params.add('b0_%i' % (iy+1), value=0, min=-np.inf, max=np.inf)
    out = minimize(objective_shared_ori_new, fit_params, args=(orient_unique, data))

    pref_ori_fit = []
    rp_fit = []
    r0_fit = []
    kappa_fit = []
    period_fit = []
    b0_fit = []
    
    for m in range(0,l):
        pref_ori_fit.append(out.params['pref_ori_%i' % (m+1)].value)
        rp_fit.append(out.params['rp_%i' % (m+1)].value)
        r0_fit.append(out.params['r0_%i' % (m+1)].value)
        kappa_fit.append(out.params['kappa_%i' % (m+1)].value)
        period_fit.append(out.params['period'].value)
        # b0_fit.append(out.params['b0'].value)
        b0_fit.append(out.params['b0_%i' % (m+1)].value)
    # print(fit_report(out.params))

    return pref_ori_fit, rp_fit, r0_fit, kappa_fit, period_fit, b0_fit

def von_mises_multi_fit_shared_A_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr):
    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = orient_unique[list(data_fr[j][dtopt_indx[j]]).index(max(list(data_fr[j][dtopt_indx[j]])))]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    data = np.array(table_x_y_expanded)
    fit_params = Parameters()
    for iy, y in enumerate(data):
        fit_params.add('pref_ori', value=pref_oris_all_units_topt, min=0, max=180)
        fit_params.add('rp_%i' % (iy+1), value=max(np.array(table_x_y_expanded).flatten()), min=np.mean(np.array(table_x_y_expanded[iy])), max=max(np.array(table_x_y_expanded).flatten()) + np.mean(np.array(table_x_y_expanded[iy])))
        fit_params.add('r0_%i' % (iy+1), value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.mean(np.array(table_x_y_expanded[iy])))
        if isNaN(tuning_width_all_units_short[iy]) == True:
            fit_params.add('kappa_%i' % (iy+1), value=0, min=0, max=10) 
        else:
            fit_params.add('kappa', value=np.sqrt(tuning_width_all_units_short[iy]), min=0, max=10)       
        fit_params.add('period', value=180, min=80, max=185)
        fit_params.add('b0_%i' % (iy+1), value=0, min=-np.inf, max=np.inf)
    out = minimize(objective_shared_new_A, fit_params, args=(orient_unique, data))
    pref_ori_fit = []
    rp_fit = []
    r0_fit = []
    kappa_fit = []
    period_fit = []
    b0_fit = []
    for m in range(0,l):

        pref_ori_fit.append(out.params['pref_ori'].value)
        rp_fit.append(out.params['rp_%i' % (m+1)].value)
        # r0_fit.append(out.params['r0'].value)
        r0_fit.append(out.params['r0_%i' % (m+1)].value)
        kappa_fit.append(out.params['kappa'].value)
        period_fit.append(out.params['period'].value)
        b0_fit.append(out.params['b0_%i' % (m+1)].value)
    return pref_ori_fit, rp_fit, r0_fit, kappa_fit, period_fit, b0_fit

def von_mises_multi_fit_shared_A_ori_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr):
    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = orient_unique[list(data_fr[j][dtopt_indx[j]]).index(max(list(data_fr[j][dtopt_indx[j]])))]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    data = np.array(table_x_y_expanded)
    fit_params = Parameters()
    for iy, y in enumerate(data):
        fit_params.add('pref_ori_%i' % (iy+1), value=pref_oris_all_units_topt, min=0, max=180)
        fit_params.add('rp_%i' % (iy+1), value=max(np.array(table_x_y_expanded).flatten()), min=np.mean(np.array(table_x_y_expanded[iy])), max=max(np.array(table_x_y_expanded).flatten()) + np.mean(np.array(table_x_y_expanded[iy])))
        fit_params.add('r0_%i' % (iy+1), value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.mean(np.array(table_x_y_expanded[iy])))
        if isNaN(tuning_width_all_units_short[iy]) == True:
            fit_params.add('kappa', value=0, min=0, max=10) 
        else:
            fit_params.add('kappa', value=np.sqrt(tuning_width_all_units_short[iy]), min=0, max=10)  
        fit_params.add('period', value=180, min=80, max=185)  
        fit_params.add('b0_%i' % (iy+1), value=0, min=-np.inf, max=np.inf)
    out = minimize(objective_shared_ori_new_A, fit_params, args=(orient_unique, data))
    pref_ori_fit = []
    rp_fit = []
    r0_fit = []
    kappa_fit = []
    period_fit = []
    b0_fit = []
    for m in range(0,l):
        pref_ori_fit.append(out.params['pref_ori_%i' % (m+1)].value)
        rp_fit.append(out.params['rp_%i' % (m+1)].value)
        # r0_fit.append(out.params['r0'].value)
        r0_fit.append(out.params['r0_%i' % (m+1)].value)
        kappa_fit.append(out.params['kappa'].value)
        period_fit.append(out.params['period'].value)
        b0_fit.append(out.params['b0_%i' % (m+1)].value)
    # print(fit_report(out.params, show_correl=False))

    return pref_ori_fit, rp_fit, r0_fit, kappa_fit, period_fit, b0_fit

def von_mises_fit_report_A_ori_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr):
    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = orient_unique[list(data_fr[j][dtopt_indx[j]]).index(max(list(data_fr[j][dtopt_indx[j]])))]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    data = np.array(table_x_y_expanded)
    fit_params = Parameters()
    for iy, y in enumerate(data):
        fit_params.add('pref_ori_%i' % (iy+1), value=pref_oris_all_units_topt, min=0, max=180)
        fit_params.add('rp_%i' % (iy+1), value=max(np.array(table_x_y_expanded).flatten()), min=np.mean(np.array(table_x_y_expanded[iy])), max=max(np.array(table_x_y_expanded).flatten()) + np.mean(np.array(table_x_y_expanded[iy])))
        fit_params.add('r0_%i' % (iy+1), value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.mean(np.array(table_x_y_expanded[iy])))
        if isNaN(tuning_width_all_units_short[iy]) == True:
            fit_params.add('kappa', value=0, min=0, max=10) 
        else:
            fit_params.add('kappa', value=np.sqrt(tuning_width_all_units_short[iy]), min=0, max=10)  
        fit_params.add('period', value=180, min=80, max=185)  
        fit_params.add('b0_%i' % (iy+1), value=0, min=-np.inf, max=np.inf)
    out = minimize(objective_shared_ori_new_A, fit_params, args=(orient_unique, data))
    pref_ori_fit = []
    rp_fit = []
    r0_fit = []
    kappa_fit = []
    period_fit = []
    b0_fit = []
    pref_ori_fit_error = []
    for m in range(0,l):

        pref_ori_fit.append(out.params['pref_ori_%i' % (m+1)].value)
        rp_fit.append(out.params['rp_%i' % (m+1)].value)
        # r0_fit.append(out.params['r0'].value)
        r0_fit.append(out.params['r0_%i' % (m+1)].value)
        kappa_fit.append(out.params['kappa'].value)
        period_fit.append(out.params['period'].value)
        b0_fit.append(out.params['b0_%i' % (m+1)].value)
        pref_ori_fit_error.append(out.params['pref_ori_%i' % (m+1)].stderr)
    # print(fit_report(out.params, show_correl=False))

    return [round(dtrange[i]*1000,2) for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)], list(zip(pref_ori_fit, pref_ori_fit_error))

#same for wrapped Gaussian

def wrapped_gauss_multi_fit_shared_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr):
    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = dtopt_indx_sorted(keys)
    pref_oris_all_units_short = [pref_oris_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = pref_oris_all_units[j][dtopt_indx[j]]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    data = np.array(table_x_y_expanded)
    fit_params = Parameters()
    for iy, y in enumerate(data):  
        fit_params.add('pref_ori', value=pref_oris_all_units_short[iy], min=0, max=180)
        fit_params.add('rp_%i' % (iy+1), value=max(np.array(table_x_y_expanded).flatten()), min=np.mean(np.array(table_x_y_expanded[iy])), max=max(np.array(table_x_y_expanded).flatten()) + np.mean(np.array(table_x_y_expanded[iy])))
        fit_params.add('r0_%i' % (iy+1), value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.mean(np.array(table_x_y_expanded[iy])))
        if tuning_width_all_units_short[iy] >180:
            fit_params.add('sigma_%i' % (iy+1), value=0, min=0, max=1)
        elif tuning_width_all_units_topt > 180:
            fit_params.add('sigma_%i' % (iy+1), value=tuning_width_all_units_short[iy], min=0, max=180)
        else:
            fit_params.add('sigma_%i' % (iy+1), value=tuning_width_all_units_topt, min=0, max=tuning_width_all_units_topt + 50)
        fit_params.add('period', value=180, min=80, max=180)
        fit_params.add('b0', value=0, min=-np.inf, max=np.inf)

    out = minimize(objective_shared_gauss, fit_params, args=(orient_unique, data))

    pref_ori_fit = []
    rp_fit = []
    r0_fit = []
    sigma_fit = []
    period_fit = []
    b0_fit = []
    
    for m in range(0,l):
        pref_ori_fit.append(out.params['pref_ori'].value)
        rp_fit.append(out.params['rp_%i' % (m+1)].value)
        r0_fit.append(out.params['r0_%i' % (m+1)].value)
        sigma_fit.append(out.params['sigma_%i' % (m+1)].value)
        period_fit.append(out.params['period'].value)
        b0_fit.append(out.params['b0'].value)
        # b0_fit.append(out.params['b0_%i' % (m+1)].value)
    return pref_ori_fit, rp_fit, r0_fit, sigma_fit, period_fit, b0_fit

def wrapped_gauss_multi_fit_shared_A_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr):
    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = dtopt_indx_sorted(keys)
    pref_oris_all_units_short = [pref_oris_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = pref_oris_all_units[j][dtopt_indx[j]]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    data = np.array(table_x_y_expanded)
    fit_params = Parameters()
    for iy, y in enumerate(data):
        fit_params.add('pref_ori', value=pref_oris_all_units_topt, min=0, max=180)
        fit_params.add('rp_%i' % (iy+1), value=max(np.array(table_x_y_expanded).flatten()), min=np.mean(np.array(table_x_y_expanded[iy])), max=max(np.array(table_x_y_expanded).flatten()) + np.mean(np.array(table_x_y_expanded[iy])))
        fit_params.add('r0', value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.mean(np.array(table_x_y_expanded[iy])))
        # fit_params.add('r0_%i' % (iy+1), value=min(np.array(table_x_y_expanded[iy])), min=0, max=np.max(np.array(table_x_y_expanded[iy])))
        fit_params.add('sigma', value=tuning_width_all_units_short[iy], min=0, max=180)
        fit_params.add('period', value=180, min=80, max=180)  
        fit_params.add('b0_%i' % (iy+1), value=0, min=-np.inf, max=np.inf)
    out = minimize(objective_shared_A_gauss, fit_params, args=(orient_unique, data))
    pref_ori_fit = []
    rp_fit = []
    r0_fit = []
    sigma_fit = []
    period_fit = []
    b0_fit = []
    for m in range(0,l):
        pref_ori_fit.append(out.params['pref_ori'].value)
        rp_fit.append(out.params['rp_%i' % (m+1)].value)
        r0_fit.append(out.params['r0'].value)
        # r0_fit.append(out.params['r0_%i' % (m+1)].value)
        sigma_fit.append(out.params['sigma'].value)
        period_fit.append(out.params['period'].value)
        b0_fit.append(out.params['b0_%i' % (m+1)].value)
    return pref_ori_fit, rp_fit, r0_fit, sigma_fit, period_fit, b0_fit

    

def plot_von_mises_fit_ori_expanded_multi_shared_new(j,keys, orient_unique,plotting,a,b, pref_oris_all_units, tuning_width_all_units,table_all,fitparams_all,data_fr,list_fr_revcorr_all):

    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    list_fr = list_fr_revcorr_all[j]
    pref_oris_all_units_short = [pref_oris_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = pref_oris_all_units[j][dtopt_indx[j]]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    if fitparams_all == True:
        fitparams_all = von_mises_multi_fit_shared_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units,data_fr)
        title_new = 'Orientation tuning with amplitude and kappa modulated von Mises approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]]

    else:
        fitparams_all = von_mises_multi_fit_shared_A_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units,data_fr)
        title_new = 'Orientation tuning with amplitude modulated von Mises approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]]

    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    pref_ori_all=fitparams_all[0]
    rp_all=fitparams_all[1]
    r0_all=fitparams_all[2]
    kappa_all=fitparams_all[3]
    period_all=fitparams_all[4]
    b0_all=fitparams_all[5]
    x_new = np.linspace(-20,180,200)
    r2_A_kappa_all = []
    res_A_kappa_all = []
    ss_res_A_kappa_all = []
    ss_tot_A_kappa_all_mean = []
    # mse_list = []
    for s in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
        
        residuals_A_kappa = []
        ss_tot_A_kappa = 0
        residuals_A_kappa = np.array(table_x_y_expanded[s]) - von_mises_new(np.array(orient_unique),  pref_ori_all[s], rp_all[s], r0_all[s], kappa_all[s], period_all[s], b0_all[s])
        ss_res_A_kappa = np.sum(residuals_A_kappa**2)
        # mse = ss_res_A_kappa/len(np.array(orient_unique))
        ss_tot_A_kappa = np.sum((np.array(table_x_y_expanded[s])-np.mean(np.array(table_x_y_expanded[s])))**2)
        ss_tot_A_kappa_mean = np.sum((np.array(table_x_y_expanded[s])-np.mean(np.array(table_x_y_expanded).flatten()))**2)
        #here we are using not a main formula r_2 = 1 - (ss_res/ss_tot) because as r^2 closer to 1, as model fit better 
        r_2_A_kappa = 1-(ss_res_A_kappa/ss_tot_A_kappa)
        r2_A_kappa_all.append(r_2_A_kappa)
        res_A_kappa_all.append(residuals_A_kappa)
        ss_res_A_kappa_all.append(ss_res_A_kappa)
        ss_tot_A_kappa_all_mean.append(ss_tot_A_kappa_mean)
        # mse_list.append(mse)
        
    # print(mse_list)

    r2_all_mean =  1-(sum(ss_res_A_kappa_all)/sum(ss_tot_A_kappa_all_mean))
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    t_list = [round(tw*1000,0) for tw in list(dtrange[dtopt_indx[j]-a:dtopt_indx[j]+b])]

    params_all = pd.DataFrame({'m': np.full(l, table_all['m'][j]), 's':np.full(l, table_all['s'][j]), 'e':np.full(l, table_all['e'][j]), 'u':np.full(l, table_all['u'][j]),'time point': t_list, 'amplitude': [fitparams_all[1][f] - fitparams_all[2][f] for f in range(0,l)],'prefered orientation': fitparams_all[0], 'max response': fitparams_all[1], 'background response': fitparams_all[2], 'kappa': fitparams_all[3], 'b0': fitparams_all[5], 'r2': r2_A_kappa_all, 'residual_ss': ss_res_A_kappa_all})
    params_all['new_col'] = range(1, len(params_all) + 1)
    params_all = params_all.groupby(['m', 's', 'e', 'u','new_col','time point'])[['amplitude', 'prefered orientation', 'max response', 'background response', 'kappa', 'b0', 'r2', 'residual_ss']].sum()
    params_all = params_all.droplevel('new_col')
    responses_list  = []
    osi_all = []
    for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
        table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
        table_x_y = table_x_y.groupby('orient')['fr'].mean()
        table_x_y = list(table_x_y)
        table_x_y.append(table_x_y[0])
        num_1 = [((list(table_x_y)[q]-min((list(table_x_y))))*np.sin(2*math.radians(orient_unique[q]))) for q in range(0, len(orient_unique))]
        num_2 = [((list(table_x_y)[q]-min((list(table_x_y))))*np.cos(2*math.radians(orient_unique[q]))) for q in range(0, len(orient_unique))]
        osi = np.sqrt(sum(num_1)**2+ sum(num_2)**2)/(sum([list(table_x_y)[q]-min((list(table_x_y))) for q in range(0, len(orient_unique))]))
        osi_all.append(osi)  
    params_all.insert(8, "OSI", osi_all) 

    # we are calculating tuning width using find_piks func to measure width of tuning curve at half hight
    results_half_all = []
    try:
        for k in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
            peaks, _ = find_peaks(von_mises_new(x_new,pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]))
            results_half = peak_widths(von_mises_new(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]), peaks, rel_height=0.5)
            results_full = peak_widths(von_mises_new(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]), peaks, rel_height=1)
            results_half_all.append(results_half[0][0])
        params_all.insert(9, "tuning width", results_half_all) 
    except:
        pass

    if plotting == True:
        a_l = np.linspace(0.6, 0.9, a)
        b_l = np.linspace(1,0.8,b)
        alpha_list = np.concatenate((a_l,b_l))
        plt.figure(figsize=(20, 8))

        # define color map
        cmap = plt.get_cmap('ocean')
        color1=mcp.gen_color(cmap="ocean",n=l+2)
        colors = np.array(range(dtopt_indx[j]-a, dtopt_indx[j]+b))

        for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
            indx_comp = w - (dtopt_indx[j]-a)
            table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
            table_x_y = table_x_y.groupby('orient')['fr'].mean()
            table_x_y = list(table_x_y)
            table_x_y.append(table_x_y[0])

            if w != dtopt_indx[j]:
                plt.scatter(orient_unique, table_x_y, label="t = %d ms" %round(dtrange[w]*1000,0), color=color1[indx_comp])
            else:
                plt.scatter(orient_unique, table_x_y, label=r'$\bf{t = %d ms}$' %round(dtrange[dtopt_indx[j]]*1000,0), color=color1[indx_comp])
            plt.xlabel('orientation, deg', fontsize = 24)
            plt.ylabel('firing rate, Hz', fontsize = 24)
            plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', prop={'size': 12})

        for k in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
            plt.plot(x_new, von_mises_new(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]),  color = color1[k])
            if k ==a:
                plt.plot(x_new, von_mises_new(x_new, pref_ori=fitparams_all[0][a], rp=fitparams_all[1][a], r0=fitparams_all[2][a], kappa=fitparams_all[3][a], period=fitparams_all[4][a], b0=fitparams_all[5][a]),  color = color1[a], linewidth=6, alpha = 0.4)
            else:
                pass

        plt.clim(vmin = dtrange[dtopt_indx[j]-a]*100, vmax = dtrange[dtopt_indx[j]+b]*100)
        seaborn.despine(top=True, right=True, left=False, bottom=False)
        plt.xticks(orient_unique, fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlim([-20,185])
        plt.show()
    else:
        return params_all
    

def plot_von_mises_fit_multi_shared_ori_new(j,keys, orient_unique,plotting,a,b, pref_oris_all_units, tuning_width_all_units,table_all,fitparams_all, data_fr, list_fr_revcorr_all):

    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    list_fr = list_fr_revcorr_all[j]
    pref_oris_all_units_short = [pref_oris_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = pref_oris_all_units[j][dtopt_indx[j]]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)

    if fitparams_all == True:
        fitparams_all = von_mises_multi_fit_shared_ori_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr)
        title_new = 'Orientation tuning with amplitude and kappa modulated von Mises approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]]

    else:
        fitparams_all = von_mises_multi_fit_shared_A_ori_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units, data_fr)
        title_new = 'Orientation tuning with amplitude modulated von Mises approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]]

    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    pref_ori_all=fitparams_all[0]
    rp_all=fitparams_all[1]
    r0_all=fitparams_all[2]
    kappa_all=fitparams_all[3]
    period_all=fitparams_all[4]
    b0_all=fitparams_all[5]
    x_new = np.linspace(0,180,180)
    r2_A_kappa_all = []
    res_A_kappa_all = []
    ss_res_A_kappa_all = []
    ss_tot_A_kappa_all_mean = []
    for s in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
        residuals_A_kappa = []
        ss_tot_A_kappa = 0
        residuals_A_kappa = np.array(table_x_y_expanded[s]) - von_mises_new(np.array(orient_unique),  pref_ori_all[s], rp_all[s], r0_all[s], kappa_all[s], period_all[s], b0_all[s])
        ss_res_A_kappa = np.sum(residuals_A_kappa**2)
        ss_tot_A_kappa = np.sum((np.array(table_x_y_expanded[s])-np.mean(np.array(table_x_y_expanded[s])))**2)
        ss_tot_A_kappa_mean = np.sum((np.array(table_x_y_expanded[s])-np.mean(np.array(table_x_y_expanded).flatten()))**2)
        #here we are using not a main formula r_2 = 1 - (ss_res/ss_tot) because as r^2 closer to 1, as model fit better 
        #we use r^2 as for error bars plotting 
        r_2_A_kappa = 1-(ss_res_A_kappa/ss_tot_A_kappa)
        r2_A_kappa_all.append(r_2_A_kappa)
        res_A_kappa_all.append(residuals_A_kappa)
        ss_res_A_kappa_all.append(ss_res_A_kappa)
        ss_tot_A_kappa_all_mean.append(ss_tot_A_kappa_mean)
    r2_all_mean =  1-(sum(ss_res_A_kappa_all)/sum(ss_tot_A_kappa_all_mean))
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    t_list = [round(tw*1000,0) for tw in list(dtrange[dtopt_indx[j]-a:dtopt_indx[j]+b])]

    params_all = pd.DataFrame({'m': np.full(l, table_all['m'][j]), 's':np.full(l, table_all['s'][j]), 'e':np.full(l, table_all['e'][j]), 'u':np.full(l, table_all['u'][j]),'time point': t_list, 'amplitude': [fitparams_all[1][f] - fitparams_all[2][f] for f in range(0,l)],'prefered orientation': fitparams_all[0], 'max response': fitparams_all[1], 'background response': fitparams_all[2], 'kappa': fitparams_all[3], 'b0': fitparams_all[5], 'r2': r2_A_kappa_all, 'residual_ss': ss_res_A_kappa_all})
    params_all['new_col'] = range(1, len(params_all) + 1)
    params_all = params_all.groupby(['m', 's', 'e', 'u','new_col','time point'])[['amplitude', 'prefered orientation', 'max response', 'background response', 'kappa', 'b0', 'r2', 'residual_ss']].sum()
    params_all = params_all.droplevel('new_col')
    responses_list  = []
    osi_all = []

    for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
        table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
        table_x_y = table_x_y.groupby('orient')['fr'].mean()
        table_x_y = list(table_x_y)
        table_x_y.append(table_x_y[0])
        num_1 = [((list(table_x_y)[q]-min((list(table_x_y))))*np.sin(2*math.radians(orient_unique[q]))) for q in range(0, len(orient_unique))]
        num_2 = [((list(table_x_y)[q]-min((list(table_x_y))))*np.cos(2*math.radians(orient_unique[q]))) for q in range(0, len(orient_unique))]
        osi = np.sqrt(sum(num_1)**2+ sum(num_2)**2)/(sum([list(table_x_y)[q]-min((list(table_x_y))) for q in range(0, len(orient_unique))]))
        osi_all.append(osi)  
    params_all.insert(8, "OSI", osi_all) 

    results_half_all = []
    try:
        for k in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
            peaks, _ = find_peaks(von_mises_new(x_new,pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]))
            results_half = peak_widths(von_mises_new(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]), peaks, rel_height=0.5)
            results_full = peak_widths(von_mises_new(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]), peaks, rel_height=1)
            results_half_all.append(results_half[0][0])
        params_all.insert(9, "tuning width", results_half_all) 
    except:
        pass

    if plotting == True:
        a_l = np.linspace(0.6, 0.9, a)
        b_l = np.linspace(1,0.8,b)
        alpha_list = np.concatenate((a_l,b_l))
        plt.figure(figsize=(20, 8))
        cmap = plt.get_cmap('ocean')
        color1=mcp.gen_color(cmap="ocean",n=l+2)
        colors = np.array(range(dtopt_indx[j]-a, dtopt_indx[j]+b))

        for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
            # uncomment line to get unnormalized firing rate
            indx_comp = w - (dtopt_indx[j]-a)
            table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
            table_x_y = table_x_y.groupby('orient')['fr'].mean()
            table_x_y = list(table_x_y)
            table_x_y.append(table_x_y[0])
            if w != dtopt_indx[j]:
                plt.scatter(orient_unique, table_x_y, label="t = %d ms" %round(dtrange[w]*1000,0), color=color1[indx_comp])
            else:
                plt.scatter(orient_unique, table_x_y, label=r'$\bf{t = %d ms}$' %round(dtrange[dtopt_indx[j]]*1000,0), color=color1[indx_comp])
            plt.xlabel('orientation, deg', fontsize = 24)
            plt.ylabel('firing rate, Hz', fontsize = 24)
            plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', prop={'size': 12})

        for k in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
            plt.plot(x_new, von_mises_new(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], kappa=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]),  color = color1[k])
            if k ==a:
                plt.plot(x_new, von_mises_new(x_new, pref_ori=fitparams_all[0][a], rp=fitparams_all[1][a], r0=fitparams_all[2][a], kappa=fitparams_all[3][a], period=fitparams_all[4][a], b0=fitparams_all[5][a]),  color = color1[a], linewidth=6, alpha = 0.4)
            else:
                pass
        
        plt.clim(vmin = dtrange[dtopt_indx[j]-a]*100, vmax = dtrange[dtopt_indx[j]+b]*100)
        seaborn.despine(top=True, right=True, left=False, bottom=False)
        plt.xticks(orient_unique, fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlim([-5,185])
        # if you want to save a plot
        # plt.savefig('Example_model2_OSI.eps', bbox_inches='tight', format='eps')
        plt.show()
    else:
        return params_all


def plot_wrapped_gauss_fit_ori_expanded_multi_shared_new(j,keys, orient_unique,plotting,a,b, pref_oris_all_units, tuning_width_all_units,table_all,fitparams_all,data_fr,list_fr_revcorr_all):

    dtrange = (RCParams & keys).fetch1('smoothdts')
    precision = RCParams().fetch1('precision')
    dtopt_indx = dtopt_indx_sorted(keys)
    list_fr = list_fr_revcorr_all[j]
    pref_oris_all_units_short = [pref_oris_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    tuning_width_all_units_short = [tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-a, dtopt_indx[j]+b)]
    pref_oris_all_units_topt = pref_oris_all_units[j][dtopt_indx[j]]
    tuning_width_all_units_topt = tuning_width_all_units[j][dtopt_indx[j]]
    table_x_y_expanded = firing_rate_loaded(j, keys, a, b, data_fr)
    if fitparams_all == True:
        fitparams_all = wrapped_gauss_multi_fit_shared_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units,data_fr)
        title_new = 'Orientation tuning with amplitude and width modulated wrapped Gaussian approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]]
    else:
        fitparams_all = wrapped_gauss_multi_fit_shared_A_new(j,keys, orient_unique,a,b, pref_oris_all_units, tuning_width_all_units,data_fr)
        title_new = 'Orientation tu15ning with amplitude modulated wrapped Gaussian approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]]

    orient_unique = list(orient_unique)
    orient_unique.append(180)
    orient_unique = np.array(orient_unique)
    pref_ori_all=fitparams_all[0]
    rp_all=fitparams_all[1]
    r0_all=fitparams_all[2]
    sigma_all=fitparams_all[3]
    period_all=fitparams_all[4]
    b0_all=fitparams_all[5]
    x_new = np.linspace(0,210,210)
    r2_A_sigma_all = []
    res_A_sigma_all = []
    ss_res_A_sigma_all = []
    ss_tot_A_sigma_all_mean = []
    for s in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
        residuals_A_sigma = []
        ss_tot_A_sigma = 0
        residuals_A_sigma = np.array(table_x_y_expanded[s]) - wrapped_gaussian_modulated(np.array(orient_unique),  pref_ori_all[s], rp_all[s], r0_all[s], sigma_all[s], period_all[s], b0_all[s])
        ss_res_A_sigma = np.sum(residuals_A_sigma**2)
        ss_tot_A_sigma = np.sum((np.array(table_x_y_expanded[s])-np.mean(np.array(table_x_y_expanded[s])))**2)
        ss_tot_A_sigma_mean = np.sum((np.array(table_x_y_expanded[s])-np.mean(np.array(table_x_y_expanded).flatten()))**2)
        #here we are using not a main formula r_2 = 1 - (ss_res/ss_tot) because as r^2 closer to 1, as model fit better 
        r_2_A_sigma = 1-(ss_res_A_sigma/ss_tot_A_sigma)
        r2_A_sigma_all.append(r_2_A_sigma)
        res_A_sigma_all.append(residuals_A_sigma)
        ss_res_A_sigma_all.append(ss_res_A_sigma)
        ss_tot_A_sigma_all_mean.append(ss_tot_A_sigma_mean)
    r2_all_mean =  1-(sum(ss_res_A_sigma_all)/sum(ss_tot_A_sigma_all_mean))
    l = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    t_list = np.arange(0, l, 1)
    params_all = pd.DataFrame({'m': np.full(l, table_all['m'][j]), 's':np.full(l, table_all['s'][j]), 'e':np.full(l, table_all['e'][j]), 'u':np.full(l, table_all['u'][j]),'time point': t_list, 'amplitude': [fitparams_all[1][f] - fitparams_all[2][f] for f in range(0,l)],'prefered orientation': fitparams_all[0], 'max response': fitparams_all[1], 'background response': fitparams_all[2], 'sigma': fitparams_all[3], 'b0': fitparams_all[5], 'r2': r2_A_sigma_all, 'residual_ss': ss_res_A_sigma_all})
    params_all['new_col'] = range(1, len(params_all) + 1)
    params_all = params_all.groupby(['m', 's', 'e', 'u','new_col','time point'])[['amplitude', 'prefered orientation', 'max response', 'background response', 'sigma', 'b0', 'r2', 'residual_ss']].sum()
    params_all = params_all.droplevel('new_col')
    responses_list  = []
    osi_all = []
    for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
        table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
        table_x_y = table_x_y.groupby('orient')['fr'].mean()
        table_x_y = list(table_x_y)
        table_x_y.append(table_x_y[0])
        num_1 = [((list(table_x_y)[q]-min((list(table_x_y))))*np.sin(2*math.radians(orient_unique[q]))) for q in range(0, len(orient_unique))]
        num_2 = [((list(table_x_y)[q]-min((list(table_x_y))))*np.cos(2*math.radians(orient_unique[q]))) for q in range(0, len(orient_unique))]
        osi = np.sqrt(sum(num_1)**2+ sum(num_2)**2)/(sum([list(table_x_y)[q]-min((list(table_x_y))) for q in range(0, len(orient_unique))]))
        osi_all.append(osi)  
    params_all.insert(8, "OSI", osi_all) 

    results_half_all = []
    try:
        for k in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
            peaks, _ = find_peaks(wrapped_gaussian_modulated(x_new,pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], sigma=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]))
            results_half = peak_widths(wrapped_gaussian_modulated(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], sigma=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]), peaks, rel_height=0.5)
            results_full = peak_widths(wrapped_gaussian_modulated(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], sigma=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]), peaks, rel_height=1)
            results_half_all.append(results_half[0][0])
        params_all.insert(9, "tuning width", results_half_all) 
    except:
        pass

    if plotting == True:
        a_l = np.linspace(0.6, 0.9, a)
        b_l = np.linspace(1,0.8,b)
        alpha_list = np.concatenate((a_l,b_l))
        plt.figure(figsize=(20, 8))
        cmap = plt.get_cmap('ocean')
        color1=mcp.gen_color(cmap="ocean",n=l+2)
        colors = np.array(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
        for w in range(dtopt_indx[j]-a, dtopt_indx[j]+b):
            indx_comp = w - (dtopt_indx[j]-a)
            table_x_y = pd.DataFrame({'orient': grat_cond_unit_j(j, keys)[1], 'fr': list_fr[w]})
            table_x_y = table_x_y.groupby('orient')['fr'].mean()
            table_x_y = list(table_x_y)
            table_x_y.append(table_x_y[0])
            plt.scatter(orient_unique, table_x_y, label="t = %d ms" %round(dtrange[w]*1000,0), color=color1[indx_comp])
            plt.xlabel('orientation, deg', fontsize = 24)
            plt.ylabel('firing rate, Hz', fontsize = 24)
            plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', prop={'size': 12})

        for k in range(0, len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))):
            plt.plot(x_new, wrapped_gaussian_modulated(x_new, pref_ori=fitparams_all[0][k], rp=fitparams_all[1][k], r0=fitparams_all[2][k], sigma=fitparams_all[3][k], period=fitparams_all[4][k], b0=fitparams_all[5][k]),  color = color1[k])
        
        plt.clim(vmin = dtrange[dtopt_indx[j]-a]*100, vmax = dtrange[dtopt_indx[j]+b]*100)
        seaborn.despine(top=True, right=True, left=False, bottom=False)
        plt.xticks(orient_unique, fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlim([-5,210])
        plt.show()
    else:
        return params_all

# STATICTICAL TEST FOR COMPARING TWO FITTING MODELS (AMPLITUDE AND AMPLITUDE+WIDTH MODULATED)

def F_test_2_models_expanded_multi_shared_ori(j,keys, orient_unique, v1, v2, a, b, pref_oris_all_units, tuning_width_all_units,table_all, data_fr, list_fr_revcorr_all):
    #v1, v2: number of parameters being estimated for models 1 and 2, respectively 
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    L = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    N = (len(orient_unique)+1)*L
    df1 = (len(orient_unique) + 1 - v1)*L
    df2 = (len(orient_unique) + 1 - v2)*L
    df_A_fit =  plot_von_mises_fit_multi_shared_ori_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,False, data_fr, list_fr_revcorr_all)
    res_ss_A_all = np.array(df_A_fit['residual_ss'])
    df_A_kappa_fit =  plot_von_mises_fit_multi_shared_ori_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,True, data_fr, list_fr_revcorr_all)
    res_ss_A_kappa_all = np.array(df_A_kappa_fit['residual_ss'])
    ss1 = np.sum(res_ss_A_all)
    ss2 = np.sum(res_ss_A_kappa_all)
    f_value = ((ss1 - ss2)/(v2*L - v1*L))/(ss2/(N-L*v2))
    p_value = 1-scipy.stats.f.cdf(f_value, df1, df2)
    return f_value, p_value

def F_test_2_models_expanded_multi_shared(j,keys, orient_unique, v1, v2, a, b, pref_oris_all_units, tuning_width_all_units,table_all, data_fr, list_fr_revcorr_all):
    #v1, v2: number of parameters being estimated for models 1 and 2, respectively 
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    L = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    N = (len(orient_unique)+1)*L
    df1 = (len(orient_unique) + 1 - v1)*L
    df2 = (len(orient_unique) + 1 - v2)*L
    df_A_fit =  plot_von_mises_fit_ori_expanded_multi_shared_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,False, data_fr, list_fr_revcorr_all)
    res_ss_A_all = np.array(df_A_fit['residual_ss'])
    df_A_kappa_fit =  plot_von_mises_fit_ori_expanded_multi_shared_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,True, data_fr, list_fr_revcorr_all)
    res_ss_A_kappa_all = np.array(df_A_kappa_fit['residual_ss'])
    ss1 = np.sum(res_ss_A_all)
    ss2 = np.sum(res_ss_A_kappa_all)
    f_value = ((ss1 - ss2)/(v2*L - v1*L))/(ss2/(N-L*v2))
    p_value = 1-scipy.stats.f.cdf(f_value, df1, df2)
    return f_value, p_value

def F_test_2_models_expanded_multi_shared_ori_amp(j,keys, orient_unique, v1, v2, a, b, pref_oris_all_units, tuning_width_all_units,table_all, data_fr, list_fr_revcorr_all):
    #v1, v2: number of parameters being estimated for models 1 and 2, respectively 
    dtopt_indx = data_fr_max(keys, data_fr)[1]
    L = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    N = (len(orient_unique)+1)*L
    df1 = (len(orient_unique) + 1 - v1)*L
    df2 = (len(orient_unique) + 1 - v2)*L
    df_A_fit =  plot_von_mises_fit_ori_expanded_multi_shared_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,False, data_fr, list_fr_revcorr_all)
    res_ss_A_all = np.array(df_A_fit['residual_ss'])
    df_A_kappa_fit =  plot_von_mises_fit_multi_shared_ori_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,False, data_fr, list_fr_revcorr_all)
    res_ss_A_kappa_all = np.array(df_A_kappa_fit['residual_ss'])
    ss1 = np.sum(res_ss_A_all)
    ss2 = np.sum(res_ss_A_kappa_all)
    f_value = ((ss1 - ss2)/(v2*L - v1*L))/(ss2/(N-L*v2))
    p_value = 1-scipy.stats.f.cdf(f_value, df1, df2)
    return f_value, p_value

def F_test_2_models_expanded_multi_shared_gauss(j,keys, orient_unique, v1, v2, a, b, pref_oris_all_units, tuning_width_all_units,table_all, data_fr, list_fr_revcorr_all):
    #v1, v2: number of parameters being estimated for models 1 and 2, respectively 
    dtopt_indx = dtopt_indx_sorted(keys)
    L = len(range(dtopt_indx[j]-a, dtopt_indx[j]+b))
    N = (len(orient_unique)+1)*L
    df1 = (len(orient_unique) + 1 - v1)*L
    df2 = (len(orient_unique) + 1 - v2)*L
    df_A_fit =  plot_wrapped_gauss_fit_ori_expanded_multi_shared_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,False, data_fr, list_fr_revcorr_all)
    res_ss_A_all = np.array(df_A_fit['residual_ss'])
    df_A_sigma_fit =  plot_wrapped_gauss_fit_ori_expanded_multi_shared_new(j,keys, orient_unique,False,a,b, pref_oris_all_units, tuning_width_all_units,table_all,True, data_fr, list_fr_revcorr_all)
    res_ss_A_sigma_all = np.array(df_A_sigma_fit['residual_ss'])
    ss1 = np.sum(res_ss_A_all)
    ss2 = np.sum(res_ss_A_sigma_all)
    f_value = ((ss1 - ss2)/(v2*L - v1*L))/(ss2/(N-L*v2))
    p_value = 1-scipy.stats.f.cdf(f_value, df1, df2)
    return f_value, p_value