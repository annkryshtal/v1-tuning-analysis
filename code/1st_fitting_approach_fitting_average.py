
# run -im djd.main


from util import c_inv_paper_series

import os
import re
import sys
import datetime
from datetime import date
from socket import getfqdn
from scipy.interpolate import interp2d

import pkg_resources 
from distutils.version import LooseVersion
from scipy.io import loadmat
import scipy
import scipy.stats as stts
from types import new_class
import numpy as np
from numpy import pi, isnan, nanmin, nanmax
import pandas as pd
import matplotlib.pyplot as plt
import datajoint as dj
import seaborn as sns
from scipy.optimize import curve_fit
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.signal import find_peaks
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

np.seterr(divide='ignore', invalid='ignore')

def def_keys(keys):
    u_keys = ((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys).fetch(dj.key)
    m_keys = ((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys).fetch('m', 's', 'e', 'u')
    unit_keys = ((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys).fetch('unit_type')
    return u_keys, m_keys, unit_keys

def def_m_keys_new(keys):
    m_keys_new = []
    m_keys = def_keys(keys)[1]
    for i in range(0, len(def_keys(keys)[0])):
        mouse = {'m':m_keys[0][i], 's': m_keys[1][i], 'e': m_keys[2][i], 'u': m_keys[3][i]}
        m_keys_new.append(mouse)
    return m_keys_new


def values_for_contrast_inv(keys):
    power_opt_all = ((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys).fetch('power_opt')
    g_z_opt_all = ((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys).fetch('g_z_opt')
    g_z_kernel_all = ((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys).fetch('g_z_kernel')
    power_s_kernel_all = ((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys).fetch('power_s_kernel')
    return power_opt_all, g_z_kernel_all, power_s_kernel_all

def df_for_contrast_inv(keys):
    df_all = pd.DataFrame((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys)
    return df_all

def def_dtopt(keys):
    m_keys_new = def_m_keys_new(keys)
    # finding optimal time point and its index in dt_range for all m_keys_new
    dtrange = (RCParams & m_keys_new).fetch1('smoothdts')

    # 'm', 's', 'e', 'u', dt_maxresponse in dtopt_df
    dtopt_df = pd.DataFrame(RCDtOpt & m_keys_new)
    dtopt_df_keys = pd.merge(pd.DataFrame((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys), dtopt_df, on=['m', 's', 'e', 'u'])
    dtopt = dtopt_df_keys['dt_maxresponse'] # array sorted by'm', 's', 'e', 'u'

    new_df_all = df_for_contrast_inv(keys).merge(dtopt_df, on=["m","s","e","u"])
    dtopt_new = new_df_all['dt_maxresponse']
    return new_df_all, dtopt_new

def e_i_unit(df):
    e_df = df.query('unit_type=="excit"')
    i_df = df.query('unit_type=="inhib"')
    return e_df, i_df

def df_exc_inh(keys):
    dtrange = (RCParams & def_m_keys_new(keys)).fetch1('smoothdts')
    e_df_all = e_i_unit(def_dtopt(keys)[0])[0]
    i_df_all = e_i_unit(def_dtopt(keys)[0])[1]
    e_df_all_index = e_df_all.index.values.tolist()
    i_df_all_index = i_df_all.index.values.tolist()
    m_keys_new_e = [def_m_keys_new(keys)[i] for i in e_df_all_index]
    m_keys_new_i = [def_m_keys_new(keys)[i] for i in i_df_all_index]
    e_df_all = e_df_all.reset_index()
    i_df_all = i_df_all.reset_index()
    dtopt_e = [def_dtopt(keys)[1][i] for i in e_df_all_index]
    dtopt_i = [def_dtopt(keys)[1][i] for i in i_df_all_index]
    dtopt_indx = []
    for i in range(0, len(def_dtopt(keys)[1])):
        dtopt_indx.append(np.where(np.isclose(dtrange, def_dtopt(keys)[1][i]))[0][0])
        
    dtopt_e_indx = []
    for i in range(0, len(dtopt_e)):
        dtopt_e_indx.append(np.where(np.isclose(dtrange, dtopt_e[i]))[0][0])

    dtopt_i_indx = []
    for i in range(0, len(dtopt_i)):
        dtopt_i_indx.append(np.where(np.isclose(dtrange, dtopt_i[i]))[0][0])

    delta_t_e = []
    for i in dtopt_e_indx:
        a = dtrange[i-5: i+6]
        delta_t_e.append(a)
        
    delta_t_i = []
    for i in dtopt_i_indx:
        a = dtrange[i-5: i+6]
        delta_t_i.append(a)  
    
    return dtopt_indx, dtopt_e_indx, dtopt_i_indx, delta_t_e, delta_t_i


def svd_vs_spatcorr_selected_t_all_units(keys, df, start, stop, ax=None, exc_color='darkorange',
                    inh_color='teal'):
    dtrange = (RCParams & def_m_keys_new(keys)).fetch1('smoothdts')
    dtopt_indx = df_exc_inh(keys)[0]
    for i in range(start, stop):
        all_df = [df.g_z_kernel[i], df.power_s_kernel[i]]
        inv_list_g_z = []
        inv_list_power = []
        inv_list_index = []
        list_power = []
        dep_list_g_z = []
        dep_list_power = []
        dep_list_index = []
        list_g_z = []
        delta_t = []
        for f in dtopt_indx:
            a = dtrange[f-5: f+6]
            delta_t.append(a)
        for j in range(dtopt_indx[i]-5, dtopt_indx[i]+6):
            if all_df[0][j]>1.96 and all_df[1][j]<=0.95:
                dep_list_g_z.append(all_df[0][j])
                dep_list_power.append(all_df[1][j])
                dep_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
            else:
                inv_list_g_z.append(all_df[0][j])
                inv_list_power.append(all_df[1][j])
                inv_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
        print(df.m[i], df.s[i], df.e[i], df.u[i],)
        print('dep_list_g_z_index = ',dep_list_index)
        print('inv_list_g_z_index = ',inv_list_index)
        print('selected g_z: ', list_g_z)
        print('selected power: ',list_power)
        print('len selected g_z: ', len(list_g_z))
        print('len selected power: ',len(list_power))
        print('dt_maxresponse = ', def_dtopt(keys)[1][i])
        unit_type = list(df.unit_type.apply(lambda x:1 if x == 'excit' else 0))
               
        if unit_type[i] == 1:
            plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=exc_color)
            plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.5)
            plt.plot(1-df.power_opt[i], df.g_z_opt[i], marker="X", markersize=10, markeredgecolor="red")
            plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
            plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                    alpha=0.5)
            text = []
            for j in range(0, len(list_power)):
                text.append(plt.text(x = 1 - list_power[j], y = list_g_z[j], s = round(delta_t[i][j],3), fontsize = 10))
            adjust_text(text, x=[1 - k for k in list_power], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
            plt.title('Fig.2(F) for selected t for all units %s' %def_m_keys_new(keys)[i])
            plt.ylabel('Pattern (g_z)')
            plt.xlabel('Residual power')
            plt.show()
        else:
            plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=inh_color)
            plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=inh_color, alpha=0.5)
            plt.plot(1-df.power_opt[i], df.g_z_opt[i], marker="X", markersize=10, markeredgecolor="red")
           
            plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
            plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                    alpha=0.5)
            text = []
            for j in range(0, len(list_power)):
                text.append(plt.text(x = 1 - list_power[j], y = list_g_z[j], s = round(delta_t[i][j],3), fontsize = 10))
            adjust_text(text, x=[1 - k for k in list_power], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
            plt.title('Fig.2(F) for selected t for all units %s' %def_m_keys_new(keys)[i])
            plt.ylabel('Pattern (g_z)')
            plt.xlabel('Residual power')
            plt.show()


def svd_vs_spatcorr_all_units(keys, df, start, stop, ax=None, exc_color='darkorange',
                    inh_color='teal'):

    dtrange = (RCParams & def_m_keys_new(keys)).fetch1('smoothdts')
    dtopt_indx = df_exc_inh(keys)[0]
    for i in range(start, stop):
        all_df = [df.g_z_kernel[i], df.power_s_kernel[i]]
        inv_list_g_z = []
        inv_list_power = []
        inv_list_index = []
        list_power = []
        dep_list_g_z = []
        dep_list_power = []
        dep_list_index = []
        list_g_z = []
        delta_t = []
        for f in dtopt_indx:
            a = dtrange[f-5: f+6]
            delta_t.append(a)
        for j in range(0, len(dtrange)):
            if all_df[0][j]>1.96 and all_df[1][j]<=0.95:
                dep_list_g_z.append(all_df[0][j])
                dep_list_power.append(all_df[1][j])
                dep_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
            else:
                inv_list_g_z.append(all_df[0][j])
                inv_list_power.append(all_df[1][j])
                inv_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
        print(df.m[i], df.s[i], df.e[i], df.u[i],)
        print('# of dep_list_g_z_index = ',len(dep_list_index))
        print('# of inv_list_g_z_index = ',len(inv_list_index))
        print('dt_maxresponse = ', def_dtopt(keys)[1][i])
        unit_type = list(df.unit_type.apply(lambda x:1 if x == 'excit' else 0))
       
        if unit_type[i] == 1:
            plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=exc_color)
            plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.5)
            plt.plot(1-df.power_opt[i], df.g_z_opt[i], marker="X", markersize=10, markeredgecolor="red")
            plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
            plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                    alpha=0.5)
            text = []
            for j in range(0, len(list_power)):
                text.append(plt.text(x = 1 - list_power[j], y = list_g_z[j], s = round(dtrange[j],3), fontsize = 10))
            adjust_text(text, x=[1 - k for k in list_power], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
            plt.title('Fig.2(F) for selected t for all units %s' %def_m_keys_new(keys)[i])
            plt.ylabel('Pattern (g_z)')
            plt.xlabel('Residual power')
            plt.show()
        else:
            plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=inh_color)
            plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=inh_color, alpha=0.5)
            plt.plot(1-df.power_opt[i], df.g_z_opt[i], marker="X", markersize=10, markeredgecolor="red")
           
            plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
            plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                    alpha=0.5)
            text = []
            for j in range(0, len(list_power)):
                text.append(plt.text(x = 1 - list_power[j], y = list_g_z[j], s = round(dtrange[j],3), fontsize = 10))
            adjust_text(text, x=[1 - k for k in list_power], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
            plt.title('Fig.2(F) for selected t for all units %s' %def_m_keys_new(keys)[i])
            plt.ylabel('Pattern (g_z)')
            plt.xlabel('Residual power')
            plt.show()

# # Firing rate calculations

# not changed 
def get_t_kernels(spike_times, trials, dtrange, precision, bin_spikes=True):
    """Collects temporal reverse correlation kernels.

    Computes the posterior probability that a stimulus elicits a spike in a
    time bin specified by precision at temporal delay delta_t for all delta_t
    in dtrange.

    Parameters
    ----------
    spike_times : np.array
        spike times
    trials : pd.DataFrame
        keys: "stim_id", "trial_on_time", "trial_off_time"
        compound representation of stimulus trials
    dtrange : np.array
        time delays at which to compute the reverse correlation
    precision : int
        decimal precision at which to bin and threshold spikes and trials
    bin_spikes : boolean
        if set to false, spikes will not be binned before kernel calculation.
        This is designed for use in resampling analyses. Setting it to False
        can result in incorrec probability values.

    Returns
    -------
    kernels : pd.DataFrame
        keys "stim_id", "delta_t", "P(Spike|Stim)"
        compound representation of spiking probability for each stimulus and
        delta_t from dtrange

    Examples
    --------
    >>> trials = pd.DataFrame({'stim_id': [0, 1, 0, 1, 0],
                               'trial_on_time': [0, 0.1, 0.2, 0.3, 0.4],
                               'trial_off_time': [0.1, 0.2, 0.3, 0.4, 0.5]})
    >>> spk_times = np.array([0.15, 0.151, 0.35, 0.45])
    >>> precision = 3
    >>> dts = np.array([-0.1, 0, 0.1])
    >>> get_t_kernels(spk_times, trials, dts, precision)
    pd.DataFrame({'stim_id': [0, 0, 0, 1, 1, 1],
                  'delta_t': [-0.1, 0, 0.1, -0.1, 0, 0.1],
                  'p_spike_stim': [3/300, 1/300, 3/300, 0/200, 3/200, 1/200]})

    References
    ----------
    Ringach, D., & Shapley, R. (2004). Reverse correlation in neurophysiology.
    Cognitive Science, 28(2), 147–166.
    https://doi.org/10.1016/j.cogsci.2003.11.003
    """
#     spike_times = spikes_times
    probs = pd.DataFrame()
    for dt in dtrange:
        Pdt = get_P_spike_given_stim(spike_times, trials, precision, dt,
                                     bin_spikes)
        probs = probs.append(Pdt, ignore_index=True)
    probs = probs.sort_values('stim_id').reset_index(drop=True)
    return probs

# not changed 
def get_P_spike_given_stim(spike_times, trials, precision, delta_t=0,
                           bin_spikes=True):
    """Calculate the posterior probability for a stimulus to elicit a spike at
    time delay delta_t (also termed reverse correlation).

    Applies Bayes' formula to spike data. The full formula is:
        P(Spike|Stim) = (P(Stim|Spike) * P(Spike)) / P(Stim)
    Which translates to:
        P(Spike|Stim) = (stim_spikes/nspikes * nspikes/nbins) /
        stimbins/nbins
    This reduces to:
        P(Spike|Stim) = stim_spikes/stimbins
    Which is what the function implements by:
        - binning and thresholding spikes at chosen precision
        - shifting spike times by delta_t
        - counting how many spike times fall into each trial
        - summing the spike counts for each stimulus (stim_spikes)
        - counting total presentation time for each stimulus in bins (stimbins)
        - dividing stim_spikes by stimbins

    Parameters
    ----------
    spike_times : np.ndarray
        spike times
    trials : pd.DataFrame
        keys: "stim_id", "trial_on_time", "trial_off_time"
        compound representation of stimulus trials
    precision : int
        decimal precision at which to bin and threshold spikes and trials
    delta_t : float
        time delay at which to compute the reverse correlation
    bin_spikes : boolean, default True
        if set to false, spikes will not be binned before kernel calculation.
        This is designed for use in resampling analyses. Setting it to False
        can result in incorrec probability values.

    Returns
    -------
    nstimspks : pd.DataFrame
        keys "stim_id", "delta_t", "P(Spike|Stim)"
        compound representation of spiking probability for each stimulus and
        the chosen delta_t

    Examples
    --------
    >>> trials = pd.DataFrame({'stim_id': [0, 1],
                               'trial_on_time': [0.0, 0.1],
                               'trial_off_time': [0.1, 0.2]})
    >>> spk_times = np.array([0.15, 0.151])
    >>> precision = 3
    >>> get_P_spike_given_stim(spk_times, trials, precision)
    pd.DataFrame({'stim_id': [0, 1],
                  'delta_t': [0, 0],
                  'p_spike_stim': [0.0, 0.02]})

    Switching off binning mode:
    >>> precision = 2
    >>> get_P_spike_given_stim(spk_times, trials, precision)
    pd.DataFrame({'stim_id': [0, 1],
                  'delta_t': [0, 0],
                  'p_spike_stim': [0.0, 0.01]})

    >>> precision = 2
    >>> get_P_spike_given_stim(spk_times, trials, precision, bin_spikes=False)
    pd.DataFrame({'stim_id': [0, 1],
                  'delta_t': [0, 0],
                  'p_spike_stim': [0.0, 0.02]})
    """
#     spike_times = spikes_times
    if bin_spikes:
        spike_times = binarize_events(spike_times, precision)
    else:
        pass
    stim_ids, spk_counts = count_stim_spikes(spike_times,
                                             trials['trial_on_time'].values,
                                             trials['trial_off_time'].values,
                                             trials['stim_id'].values,
                                             delta_t)
    nstimspks = pd.DataFrame({'stim_id': stim_ids,
                              'delta_t': np.repeat(delta_t, len(stim_ids)),
                              'p_spike_stim': spk_counts})
    nstimbins = count_stim_bins(trials['stim_id'].values,
                                trials['trial_on_time'].values,
                                trials['trial_off_time'].values,
                                precision)
    nstimspks['p_spike_stim'] = nstimspks['p_spike_stim']/nstimbins['bins']
    return nstimspks

# not changed 
def binarize_events(event_times, precision):
    """Bin event times at the specified precision level (number of decimals).

    Rounds event times to specified precision level and then keeps only unique
    event times (equivalent to one event per bin).

    Parameters
    ----------
    event_times : array_like
        event times
    precision : int
        number of decimals

    Returns
    -------
    res : ndarray
        array of binned event times

    Examples
    --------
    >>> evnt_times = np.array([1.21, 1.22, 2.5])
    >>> binarize_events(evnt_times, 1)
    array([1.2, 2.5])

    >>> evnt_times = np.array([1.215, 1.223, 2.5])
    >>> binarize_events(evnt_times, 2)
    array([1.22, 2.5])
    """
    correct_input_type = type(precision) == int or type(precision) == np.int64
    if not correct_input_type:
        raise ValueError('arg "precision" dtype is supposed to be float.')
    int_evnt_times = np.round(event_times*10**precision).astype(int)
    return np.unique(int_evnt_times)/10**precision

# not changed 
def count_stim_spikes(spike_times, trial_on_times, trial_off_times, stim_ids, delta_t=0):
    """Count how many times a particular stimulus was displayed delta_t before
    a spike occurred.

    Parameters
    ----------
    spk_times : np.ndarray
        Time points at which a spike occured.
    trial_on_times : np.ndarray
        Trial start times
    trial_off_times : np.ndarray
        Trial stop times
    delta_t : float, optional
        Investigated difference between spike and stimulus presentation.

    Returns
    -------
    overall_stims : np.ndarray
        Unique stimulus ID that occurred in the trials.
    counts_spiking_stims : np.ndarray
        Number of spikes occurring delta_t after the stimulus presentation

    Notes
    -----
    All times have to be entered in the same format (e.g. milliseconds)

    Examples
    --------
    >>> trial_on_times = np.array([0, 0.21])
    >>> trial_off_times = np.array([0.1, 0.3])
    >>> stim_ids = np.array([1, 3])
    >>> spikes = np.array([0.01, 0.25, 0.29])
    >>> count_stim_spikes(spikes, trial_on_times, trial_off_times, stim_ids)
    ([1, 3], [1, 2])
    """
#     spk_times = spikes_times
    spk_times = spike_times - delta_t
    # make sure trials are in ascending temporal order
    order = np.argsort(trial_on_times)
    trial_on_times = trial_on_times[order]
    trial_off_times = trial_off_times[order]
    stim_ids = stim_ids[order]
    # extract stimulus counts
    trial_indices = event_in_interval(spk_times, trial_on_times,
                                      trial_off_times)
    stimuli_preceeding_spikes = stim_ids[trial_indices]
    overall_stims = np.unique(stim_ids)
    uniques = np.unique(stimuli_preceeding_spikes, return_counts=True)
    (spiking_stims, tmp_counts_spiking_stims) = uniques
    # add zeros for stimuli that did not elicit any spikes
    spiking_stims_idcs = np.searchsorted(overall_stims, spiking_stims)
    counts_spiking_stims = np.zeros(len(overall_stims)).astype(int)
    counts_spiking_stims[spiking_stims_idcs] = tmp_counts_spiking_stims
    return overall_stims, counts_spiking_stims

# not changed 
def one_event_in_interval(on_off, event_t):
    """on_off: list of intervals
    lower border <eventtime <= upperborder
    """
    on, off = on_off.T
    evnt_position_on = np.searchsorted(on, event_t)
    evnt_position_off = np.searchsorted(off, event_t)
    # if event falls within interval, the on index must be larger than the
    # corresponding off index. The off index then gives the proper stimulus id
    evnt_in_interval = (evnt_position_on != evnt_position_off)
    return evnt_in_interval


# not changed 
def event_in_interval(event_times, interval_on_times, interval_off_times, precision=6,
                      return_event_times=False):
    """Check if events occured during each intervall and returns the indices of
    intervals with events.

    Parameters
    ----------
    event_times : array_like
        Time points at which events occured.
    interval_on_times : array_like
        Time points at which intervals started.
    interval_off_times : array_like
        Time points at which intervals ended.
    precision : int
        Decimal precision that will be used. Its purpose is  to avoid ambiguity
        due to float precision math.
    return_event_times : bool
        Flag to return the event times corresponding to the intervals

    Returns
    -------
    res : np.ndarray
        Indices of intervals where spikes occurred. If one interval had
        multiple spikes, its index will be listed multiple times.

    Examples
    --------
    >>> event_times = [0.5]
    >>> interval_on_times = [0]
    >>> interval_off_times = [1]
    >>> event_in_interval(event_times, interval_on_times, interval_off_times)
    array([0])

    >>> event_times = [0.5, 0.6]
    >>> interval_on_times = [0]
    >>> interval_off_times = [1]
    >>> event_in_interval(event_times, interval_on_times, interval_off_times)
    array([0, 0])

    >>> event_times = [0.5, 0.6, 1.2]
    >>> interval_on_times = [0, 1]
    >>> interval_off_times = [1, 2]
    >>> event_in_interval(event_times, interval_on_times, interval_off_times)
    array([0, 0, 1])
    """
    assert type(event_times) in (list, np.ndarray, pd.Series)
    assert type(interval_on_times) in (list, np.ndarray, pd.Series)
    assert type(interval_off_times) in (list, np.ndarray, pd.Series)
    assert type(precision) == int
    # rounding values to prevent float comparison errors
    interval_on_times = np.round(interval_on_times, precision)
    interval_off_times = np.round(interval_off_times, precision)
    event_times = np.round(event_times, precision)
    evnt_position_on = np.searchsorted(interval_on_times, event_times)
    evnt_position_off = np.searchsorted(interval_off_times, event_times)
    # if spike falls within interval, the on index must be larger than the
    # corresponding off index. The off index then gives the proper stimulus id
    evnt_in_interval = evnt_position_on != evnt_position_off
    if not return_event_times:
        return evnt_position_off[evnt_in_interval]
    else:
        return evnt_position_off[evnt_in_interval], event_times[evnt_in_interval]

# not changed 
def count_stim_bins(stim_ids, trial_on_times, trial_off_times, precision):
    """Discretises cumulative stimulus presentation time by counting the number of
    stimulus presentation bins.

    Parameters
    ----------
    stim_ids : np.ndarray
    trial_on_times : np.ndarray
    trial_off_times : np.ndarray
    precision : int

    Returns
    -------
    stimbins : pd.DataFrame with keys "stim_id" and "bins"

    Examples
    --------
    >>> stim_ids = np.array([0, 1, 1])
    >>> trial_on_times = np.array([0.01, 0.03, 0.05])
    >>> trial_off_times = np.array([0.02, 0.04, 0.06])
    >>> precision = 3
    >>> count_stim_bins(stim_ids, trial_on_times, trial_off_times, precision)
    pd.DataFrame({'stim_id': [0, 1],
                  'bins': [10, 20]})
    """
    stimbins = pd.DataFrame({'stim_id': stim_ids})
    stimbins['bins'] = (trial_off_times-trial_on_times) * 10**precision
    stimbins = stimbins.groupby('stim_id').sum().reset_index()
    stimbins = stimbins.round({'bins': 0})
    stimbins = stimbins.astype({'bins': int})
    return stimbins



def df_firing_rate(keys):
    m_keys_new = def_m_keys_new(keys)
    dfff_e_i = Unit().Spikes & (Stimulus() & m_keys_new) & m_keys_new
    spk_times_all_units = (Unit().Spikes & (Stimulus() & m_keys_new) & m_keys_new).fetch('spk_times')

    list_keys_new_e_i = []
    for i in range(0, len(m_keys_new)):
        mouse = {'m':dfff_e_i.fetch('m')[i], 's': dfff_e_i.fetch('s')[i], 'e': dfff_e_i.fetch('e')[i], 'u': dfff_e_i.fetch('u')[i]}
        list_keys_new_e_i.append(mouse)
    
    trials_all_units = []
    for i in list_keys_new_e_i:
        trial = (Stimulus.Trial & i).df
        trials_all_units.append(trial)

    dtrange = RCParams().fetch1('smoothdts')
    precision = RCParams().fetch1('precision')

    kernels_all_units = []
    for i in range(0,len(m_keys_new)):
        kernel = get_t_kernels(spk_times_all_units[i], trials_all_units[i], dtrange, precision, bin_spikes=True)
        kernels_all_units.append(kernel)

    firing_rates_df_all_units = kernels_all_units
    firing_rates_df_all_units = [firing_rates_df_all_units[i].rename(columns={'p_spike_stim': 'firing_rates'}) for i in range(0, len(m_keys_new))]

    for i in firing_rates_df_all_units:
        i['firing_rates'] = i['firing_rates'].apply(lambda x: x/(0.001))
        i['firing_rates_normalized'] = (i['firing_rates']-i.firing_rates.min())/(i.firing_rates.max()-i.firing_rates.min())

    delta_t_all_units = [firing_rates_df_all_units[i].groupby('stim_id')['delta_t'].apply(list) for i in range(0,len(m_keys_new))]
    fr_all_units = [firing_rates_df_all_units[i].groupby('stim_id')['firing_rates'].apply(list) for i in range(0,len(m_keys_new))]
    fr_norm_all_units = [firing_rates_df_all_units[i].groupby('stim_id')['firing_rates_normalized'].apply(list) for i in range(0,len(m_keys_new))]

    fr_list_all_units = []
    fr_list_norm_all_units = []
    delta_t_list_all_units = []
    for i in range(0,len(m_keys_new)):
        fr_list = []
        fr_list_norm = []
        delta_t_list = []
        for j in range(0, firing_rates_df_all_units[i].stim_id.max() + 1):
            fr_list.append(fr_all_units[i][j])
            fr_list_norm.append(fr_norm_all_units[i][j])
            delta_t_list.append(delta_t_all_units[i][j])
        fr_list_all_units.append(fr_list)
        fr_list_norm_all_units.append(fr_list_norm)
        delta_t_list_all_units.append(delta_t_list)
    return list_keys_new_e_i, firing_rates_df_all_units, delta_t_list_all_units, fr_list_all_units, fr_list_norm_all_units






def df_grat_cond_all(keys):
    m_keys_new = def_m_keys_new(keys)
    grat_cond_all= []
    grat_cond_df_all= []
    

    for i in range(0,len(m_keys_new)):
        grat_cond_all.append((Stimulus.GratingCond & df_firing_rate(keys)[0][i]).fetch('stim_id', 'grat_orientation', 'grat_contrast'))
        df = pd.DataFrame({'stim_id':grat_cond_all[i][0], 'grat_orientation':grat_cond_all[i][1], 'grat_contrast':grat_cond_all[i][2]})
        df['stim_id'] = df['stim_id']
        df.set_index('stim_id', append=True, inplace=True)
        df = df.droplevel(level = 0)
        grat_cond_df_all.append(df)
    return grat_cond_all, grat_cond_df_all

def list_fr_fr_norm(keys):
    dtrange = (RCParams & def_m_keys_new(keys)).fetch1('smoothdts')
    m_keys_new = def_m_keys_new(keys)
    firing_rates_df_all_units = df_firing_rate(keys)[1]
    delta_t_list_all_units = df_firing_rate(keys)[2]
    fr_list_all_units = df_firing_rate(keys)[3]
    fr_list_norm_all_units = df_firing_rate(keys)[4]
    list_fr_all_units = []
    for i in range(0,len(m_keys_new)):
        list_fr = []
        for j in range(0, df_firing_rate(keys)[1][i].stim_id.max() + 1):
            list_fr_short = list(zip(df_firing_rate(keys)[2][i][j], df_firing_rate(keys)[3][i][j]))
            list_fr_short = sorted(list_fr_short, key = lambda x: x[0])
            list_fr.append(list_fr_short)
        list_fr_all_units.append(list_fr)


    list_fr_all_units_norm = []
    for i in range(0,len(m_keys_new)):
        list_fr_norm = []
        for j in range(0, firing_rates_df_all_units[i].stim_id.max() + 1):
            list_fr_short_norm = list(zip(delta_t_list_all_units[i][j], fr_list_norm_all_units[i][j]))
            list_fr_short_norm = sorted(list_fr_short_norm, key = lambda x: x[0])
            list_fr_norm.append(list_fr_short_norm)
        list_fr_all_units_norm.append(list_fr_norm)


    list_full_fr_all_units = []
    for k in range(0,len(m_keys_new)):
        list_full_fr = []
        for j in range(0, len(dtrange)):
            list_full_fr_all_units_short = []
            for i in range(0, firing_rates_df_all_units[k].stim_id.max() + 1):
                list_full_fr_all_units_short.append(list_fr_all_units[k][i][j][1])
            list_full_fr.append(list_full_fr_all_units_short)
        list_full_fr_all_units.append(list_full_fr)



    list_full_fr_norm_all_units = []
    for k in range(0,len(m_keys_new)):
        list_full_fr_norm = []
        for j in range(0, len(dtrange)):
            list_full_fr_all_units_short_norm = []
            for i in range(0, firing_rates_df_all_units[k].stim_id.max() + 1):
                list_full_fr_all_units_short_norm.append(list_fr_all_units_norm[k][i][j][1])
            list_full_fr_norm.append(list_full_fr_all_units_short_norm)
        list_full_fr_norm_all_units.append(list_full_fr_norm)
    
    return list_full_fr_all_units, list_full_fr_norm_all_units

def unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    x = []
    for i in unique_list:
        x.append(i)
    x = sorted(x)
    return x


def orient_unique(keys):
    return unique(list(df_grat_cond_all(keys)[0][1][1]))

def contr_unique(keys):
    return unique(list(df_grat_cond_all(keys)[0][1][2]))


# ## Plotting extraplots (e.g. g_z(t)) ##

def svd_exc_g_z_selected_t(df, dtopt_e_indx, delta_t_e, dtopt_e, m_keys_new_e, ax=None, exc_color='darkorange',
                    inh_color='teal'):


    for i in range(0, len(df)):

        # inv_df = df.query('g_z_kernel[j]<=1.96 or power_s_kernel[j]>0.95')
        # dep_df = df.query('g_z_kernel[j]>1.96 and power_s_kernel[j]<=0.95')
        all_df = [df.g_z_kernel[i], df.power_s_kernel[i]]
        inv_list_g_z = []
        inv_list_power = []
        inv_list_index = []
        list_power = []
        dep_list_g_z = []
        dep_list_power = []
        dep_list_index = []
        list_g_z = []
   
        for j in range(dtopt_e_indx[i]-5, dtopt_e_indx[i]+6):
            if all_df[0][j]>1.96 and all_df[1][j]<=0.95:
                dep_list_g_z.append(all_df[0][j])
                dep_list_power.append(all_df[1][j])
                dep_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
            else:
                inv_list_g_z.append(all_df[0][j])
                inv_list_power.append(all_df[1][j])
                inv_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
        
       
        plt.scatter(list(delta_t_e[i]), list_g_z, color=exc_color)
        # plt.scatter(delta_t_e[i], dep_list_g_z, color=exc_color, alpha=0.5)
        # plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.5) 
        plt.plot(dtopt_e[i], df.g_z_opt[i], marker="X", markersize=10, markeredgecolor="red")
        # plt.plot([1 - j for j in inv_list_power], inv_list_g_z, color=exc_color, alpha = 0.5)
        # plt.plot([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.2)
        plt.hlines(1.96, min(delta_t_e[i]) - 0.01, max(delta_t_e[i]) + 0.01 , linestyles='dashed', alpha=0.5)
        # plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
        #         alpha=0.5)
        text = []
        for j in range(0, len(list_g_z)):
            text.append(plt.text(x = delta_t_e[i][j], y = list_g_z[j], s = round(delta_t_e[i][j],3), fontsize = 10))
        adjust_text(text, x=delta_t_e[i], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
        plt.title('Fig.2(F) for all t for exc %s' %m_keys_new_e[i])
        plt.ylabel('Pattern (g_z)')
        plt.xlabel('Time point')
        # plt.savefig('Fig_2_F for selected t for exc_u %s .png' %m_keys_new_e[i])
        plt.show()
        # plt.clf()


def svd_exc_g_z_selected_t_u(df, dtopt_e_indx, a, num_plots_, ax=None, exc_color='darkorange',
                    inh_color='teal'):
    plt.figure(figsize=(15, 5))


    colormap = plt.cm.coolwarm
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots_))))
    
    for i in a:

        # inv_df = df.query('g_z_kernel[j]<=1.96 or power_s_kernel[j]>0.95')
        # dep_df = df.query('g_z_kernel[j]>1.96 and power_s_kernel[j]<=0.95')
        all_df = [df.g_z_kernel[i], df.power_s_kernel[i]]
        inv_list_g_z = []
        inv_list_power = []
        inv_list_index = []
        list_power = []
        dep_list_g_z = []
        dep_list_power = []
        dep_list_index = []
        list_g_z = []
   
        # for j in range(dtopt_e_indx[i]-5, dtopt_e_indx[i]+6):
        for j in range(dtopt_e_indx[df['level_0'][i]]-5, dtopt_e_indx[df['level_0'][i]]+6): 
            if all_df[0][j]>1.96 and all_df[1][j]<=0.95:
                dep_list_g_z.append(all_df[0][j])
                dep_list_power.append(all_df[1][j])
                dep_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
            else:
                inv_list_g_z.append(all_df[0][j])
                inv_list_power.append(all_df[1][j])
                inv_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
        
        time = ['t1', 't2', 't3', 't4', 't5', 't_opt', 't6', 't7' , 't8', 't9', 't10']
        # plt.scatter(list(delta_t_e[i]), list_g_z)
        # plt.plot(list(delta_t_e[i]), list_g_z, '-o', alpha = 0.6, label='%s' %m_keys_new_e[i])
        plt.plot(time, list_g_z, '-o', color = "black", alpha = 0.6, label='%s' %m_keys_new_e[i])
        # plt.scatter(delta_t_e[i], dep_list_g_z, color=exc_color, alpha=0.5)
        # plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.5) 
        # plt.plot(dtopt_e[i], df.g_z_opt[i], marker="X", markersize=10, markeredgecolor="red", markerfacecolor = "red")
        # plt.plot(time[5], df.g_z_opt[i], marker="X", markersize=10, markeredgecolor="red", markerfacecolor = "red")
        # plt.plot([1 - j for j in inv_list_power], inv_list_g_z, color=exc_color, alpha = 0.5)
        # plt.plot([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.2)
        # plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', prop={'size': 8.5})
    plt.hlines(1.96, time[0], time[-1], linestyles='dashed', alpha=1, linewidth=5, zorder = 10000, color = "y")
    plt.vlines(time[5], -4, 10, colors='r', linestyles='solid', label='', zorder=10000)
    # plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
    #         alpha=0.5)
    # text = []
    # for j in range(0, len(list_g_z)):
    #     text.append(plt.text(x = delta_t_e[i][j], y = list_g_z[j], s = round(delta_t_e[i][j],3), fontsize = 10))
    # adjust_text(text, x=delta_t_e[i], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
    plt.title('g_z(t) for selected time points and units')
    plt.ylabel('Pattern (g_z)')
    plt.xlabel('Time point, s')
    # plt.xticks(dtrange[3:21])

    # plt.savefig('Fig_2_F for selected t for exc_u %s .png' %m_keys_new_e[i])
    plt.show()
        # plt.clf()


def svd_exc_power_selected_t_u(df, dtopt_e_indx, a, num_plots_, ax=None, exc_color='darkorange',
                    inh_color='teal'):
    plt.figure(figsize=(15, 5))
    colormap = plt.cm.coolwarm
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots_))))
    
    for i in a:

        # inv_df = df.query('g_z_kernel[j]<=1.96 or power_s_kernel[j]>0.95')
        # dep_df = df.query('g_z_kernel[j]>1.96 and power_s_kernel[j]<=0.95')
        all_df = [df.g_z_kernel[i], df.power_s_kernel[i]]
        inv_list_g_z = []
        inv_list_power = []
        inv_list_index = []
        list_power = []
        dep_list_g_z = []
        dep_list_power = []
        dep_list_index = []
        list_g_z = []
   
        # for j in range(dtopt_e_indx[i]-5, dtopt_e_indx[i]+6):
        for j in range(dtopt_e_indx[df['level_0'][i]]-5, dtopt_e_indx[df['level_0'][i]]+6): 
            if all_df[0][j]>1.96 and all_df[1][j]<=0.95:
                dep_list_g_z.append(all_df[0][j])
                dep_list_power.append(all_df[1][j])
                dep_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
            else:
                inv_list_g_z.append(all_df[0][j])
                inv_list_power.append(all_df[1][j])
                inv_list_index.append(j)
                list_g_z.append(all_df[0][j])
                list_power.append(all_df[1][j])
        
        time = ['t1', 't2', 't3', 't4', 't5', 't_opt', 't6', 't7' , 't8', 't9', 't10']
        # plt.scatter(list(delta_t_e[i]), list_g_z)
        plt.plot(time, [1 - j for j in list_power], '-o', color = "black", alpha = 0.6, label='%s' %m_keys_new_e[i])
        # plt.scatter(delta_t_e[i], dep_list_g_z, color=exc_color, alpha=0.5)
        # plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.5) 
        # plt.plot(time[5], 1 - df.power_opt[i], marker="X", markersize=10, markeredgecolor="red", markerfacecolor = "red")
        # plt.plot([1 - j for j in inv_list_power], inv_list_g_z, color=exc_color, alpha = 0.5)
        # plt.plot([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.2)
        # plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', prop={'size': 8.5})
    plt.hlines(0.05, time[0], time[-1], linestyles='dashed', alpha=1, linewidth = 5, zorder=10000, color = "y")
    plt.vlines(time[5], 0, 1, colors='r', linestyles='solid', label='', zorder=10000)
    # plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
    #         alpha=0.5)
    # text = []
    # for j in range(0, len(list_g_z)):
    #     text.append(plt.text(x = delta_t_e[i][j], y = list_g_z[j], s = round(delta_t_e[i][j],3), fontsize = 10))
    # adjust_text(text, x=delta_t_e[i], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
    plt.title('residual_power(t) for selected time points and units')
    plt.ylabel('Residual power')
    plt.xlabel('Time point, s')
    # plt.xticks(dtrange[3:21])

    # plt.savefig('Fig_2_F for selected t for exc_u %s .png' %m_keys_new_e[i])
    plt.show()
        # plt.clf()




# ## Tuning curves for orientation


# valid_fits= RCTun.OptWrapGauss & 'r2>0.4' & m_keys_new & (RCSVD * RCSpatialAuto.Error &
#               '(g_z_opt<=1.96 or power_opt>0.95) or (g_z_opt<=1.96 and power_opt<=0.95) or (g_z_opt>1.96 and power_opt>0.95)')
# tuning_width = valid_fits.fetch('tuning_width')



# table_all = pd.DataFrame(RCTun.WrapGauss & m_keys_new)


# valid_fits_df = pd.DataFrame(valid_fits)



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
        for i in range(k):
            res += np.exp(-(x - pref_ori + period * (i + 1)) ** 2 / (2 * sigma ** 2))
            res += np.exp(-(x - pref_ori - period * (i + 1)) ** 2 / (2 * sigma ** 2))
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


def wrapped_gaussian_modulated(x, A=1, pref_ori=90, rp=1, r0=0, sigma=30, period=180, n=None):
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
    A : amplitude
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
        for i in range(k):
            res += np.exp(-(x - pref_ori + period * (i + 1)) ** 2 / (2 * sigma ** 2))
            res += np.exp(-(x - pref_ori - period * (i + 1)) ** 2 / (2 * sigma ** 2))
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
    y = A*_wg(n)
    return y



def population_tuning(tuning_widths, ax=None, color='darkorange'):
    if ax is None:
        fig, ax = plt.subplots()

    oris = np.linspace(0, 180, 100)
    labeloris = np.linspace(-90, 90, 100)
    for tuning_width in tuning_widths:
        ax.plot(labeloris, wrapped_gaussian(oris, sigma=tuning_width), color=color, alpha=0.2)
    ax.set_xlabel('Orientation ($^{\circ}$)')
    ax.set_ylabel('Response')
    ax.spines['left'].set_bounds(ax.dataLim.ymin, ax.dataLim.ymax)
    ax.spines['bottom'].set_bounds(ax.dataLim.xmin, ax.dataLim.xmax)


def plot_wrapp_gauss_unit(j, grat_cond_all_e, list_full_fr_all_e_units, dtrange, table_all):
    plt.figure(figsize=(16, 8))
    num_plots = 11
    colormap = plt.cm.coolwarm
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

    for i in range(8,13):
        table_x_y = pd.DataFrame({'orient': grat_cond_all_e[j][1], 'fr': list_full_fr_all_e_units[j][i]})
        table_x_y = table_x_y.groupby('orient')['fr'].mean()
        table_x_y_new_all_e_units  = [(j-min(list(table_x_y)))/(max(list(table_x_y))-min(list(table_x_y))) for j in list(table_x_y)]
        x_new = np.linspace(0, 180,500)
        plt.scatter(orient_unique, table_x_y, label="t = %d ms" %round(dtrange[i]*1000,0))
        plt.xlabel('orientation, deg')
        plt.ylabel('firing rate, Hz')
        plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', prop={'size': 12})

    plt.title('Normalized orientation tuning with wrapped gaussian approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]])
    plt.show()


def po_all_tw_all_lists(m_keys_new, dtopt_indx, grat_cond_all, list_full_fr_all_units, table_all):
    table_x_y_all_units = []
    for j in range(0,len(m_keys_new)):
        table_x_y_all= []
        for i in range(dtopt_indx[j]-2, dtopt_indx[j]+3):
            table_x_y = pd.DataFrame({'orient': grat_cond_all[j][1], 'fr': list_full_fr_all_units[j][i]})
            table_x_y = table_x_y.groupby('orient')['fr'].mean()
            table_x_y_all.append(list(table_x_y))
        table_x_y_all_units.append(table_x_y_all)


    pref_oris_all_units = list(table_all.pref_ori_kernel)
    tuning_width_all_units = list(table_all.tuning_width_kernel)
    return(pref_oris_all_units, tuning_width_all_units)


def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t  
    avg = sum_num / len(num)
    return avg        
 


def table_x_y_all_units_mean(m_keys_new, table_x_y_all_units):
    table_x_y_all_units_mean = []
    for k in range(0,len(m_keys_new)):
        a= []
        for i in range(0,12):
            num = [table_x_y_all_units[k][j][i] for j in range(5)]
            a.append(cal_average(num))
        table_x_y_all_units_mean.append(a)
    return(table_x_y_all_units_mean)


def po_tw_all_units_mean(m_keys_new, pref_oris_all_units, dtopt_indx, tuning_width_all_units):
    pref_oris_all_units_mean = []
    tuning_width_all_units_mean = []

    for j in range(0,len(m_keys_new)):
        pref_oris_all_units_mean.append(cal_average([pref_oris_all_units[j][i] for i in range(dtopt_indx[j]-2, dtopt_indx[j]+3)]))
        tuning_width_all_units_mean.append(cal_average([tuning_width_all_units[j][i] for i in range(dtopt_indx[j]-2, dtopt_indx[j]+3)]))
    return pref_oris_all_units_mean, tuning_width_all_units_mean







def plot_wrapped_gauss_mean_ori(j,orient_unique,pl, table_all, pref_oris_all_units_mean, tuning_width_all_units_mean):

    x_new = np.linspace(-20, 180,500)
    orient_unique_all = [int(i) for i in orient_unique]
    fitparams_all, _ = curve_fit(wrapped_gaussian, np.array(orient_unique_all), np.array(table_x_y_all_units_mean[j]), p0 = (pref_oris_all_units_mean[j], 5, 0.2, tuning_width_all_units_mean[j], 180), bounds=((0, 0, 0, 0, 0), (180, np.inf, np.inf, 180, 180)))

    if pl == 1:
        plt.title('Orientation tuning with wrapped gaussian approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]])
        plt.scatter(orient_unique, table_x_y_all_units_mean[j])
        plt.plot(x_new, wrapped_gaussian(x_new, pref_ori=fitparams_all[0], rp=fitparams_all[1], r0=fitparams_all[2], sigma=fitparams_all[3], period=fitparams_all[4], n=None))
        plt.show()
    else:
        return fitparams_all, j



fitparams_all = plot_wrapped_gauss_mean_ori(69, orient_unique,0)


orient_unique = [int(i) for i in orient_unique]



def plot_wrapped_gauss_fit_ori(j,orient_unique,plotting, table_x_y_all_units, dtopt_indx, grat_cond_all, list_full_fr_all_units, dtrange, table_all):
    fitparams_all = plot_wrapped_gauss_mean_ori(j,orient_unique,0)
    pref_ori_all=fitparams_all[0][0]
    rp_all=fitparams_all[0][1]
    r0_all=fitparams_all[0][2]
    sigma_all=fitparams_all[0][3]
    period_all=fitparams_all[0][4]
    A_all = 1
    n=None
    
    fit_A_sigma_all = []
    for s in range(0,5):

        eps = 10**(-12)
        fitparams_A_sigma_all, _ = curve_fit(wrapped_gaussian_modulated, np.array(orient_unique), np.array(table_x_y_all_units[j][s]), p0 = (A_all, pref_ori_all, rp_all, r0_all, sigma_all, period_all), bounds=((-np.inf, pref_ori_all - eps, rp_all - eps, r0_all - eps, 0, period_all-eps), (np.inf, pref_ori_all + eps, rp_all +eps, r0_all + eps, 180, period_all+eps)))
        fit_A_sigma_all.append(fitparams_A_sigma_all)

    t_list = ['t_opt - 2Δt', 't_opt - Δt', 't_opt', 't_opt + Δt', 't_opt + 2Δt']
    A_list_all = []
    sigma_list_all = []

    for p in fit_A_sigma_all:
        A_list_all.append(p[0])
        sigma_list_all.append(p[4])

    params_all = pd.DataFrame({'m': np.full(5, table_all['m'][j]), 's':np.full(5, table_all['s'][j]), 'e':np.full(5, table_all['e'][j]), 'u':np.full(5, table_all['u'][j]),'time point': t_list, 'amplitude': A_list_all, 'prefered orientation': fitparams_all[0][0], 'max response': fitparams_all[0][1], 'background response': fitparams_all[0][2], 'tuning width': sigma_list_all})
    params_all['new_col'] = range(1, len(params_all) + 1)
    params_all = params_all.groupby(['m', 's', 'e', 'u','new_col','time point'])[['amplitude', 'prefered orientation', 'max response', 'background response', 'tuning width']].sum()
    params_all = params_all.droplevel('new_col')

    if plotting == True:
        plt.figure(figsize=(16, 8))
        for i in range(dtopt_indx[j]-2, dtopt_indx[j]+3):
            table_x_y = pd.DataFrame({'orient': grat_cond_all[j][1], 'fr': list_full_fr_all_units[j][i]})
            table_x_y = table_x_y.groupby('orient')['fr'].mean()    
            x_new = np.linspace(-15, 180,500)
            plt.scatter(orient_unique, table_x_y, label="t = %d ms" %round(dtrange[i]*1000,0))
            plt.xlabel('orientation, deg')
            plt.ylabel('firing rate, Hz')
            plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', prop={'size': 12})

        for k in range(0,5):
            plt.plot(x_new, wrapped_gaussian_modulated(x_new, A = A_list_all[k], pref_ori=fitparams_all[0][0], rp=fitparams_all[0][1], r0=fitparams_all[0][2], sigma=sigma_list_all[k], period=fitparams_all[0][4], n=None))
        plt.title('Orientation tuning with amplitude and sigma modulated wrapped gaussian approximation %s' %[table_all['m'][j],table_all['s'][j], table_all['e'][j],table_all['u'][j]], fontsize=14)
        plt.show()
    else:

        return params_all


