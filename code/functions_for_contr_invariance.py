from util import c_inv_paper_series

import os
import re
import sys
import datetime
from datetime import date
from socket import getfqdn
from scipy.interpolate import interp2d
from functions_for_fitting_updated import *

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
import datajoint as dj
import seaborn as sns
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

# svd_vs_spatcorr_selected_t_all_units(keys, df_all, 0, 10, ax=None, exc_color='darkorange',inh_color='teal')

def svd_exc_g_z_selected_t_u(df, a, num_plots_, ax=None, exc_color='darkorange',
                    inh_color='teal'):
    
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

def svd_vs_spatcorr_all_inh(keys, indexes, dtrange, dtopt_indx, ax=None, exc_color='darkorange',
                    inh_color='teal'):

    plt.figure(figsize=(6, 4))
    df_all = df_for_contrast_inv(list_keys_new_e_i(m_keys_new)).sort_values(by = ['m', 's', 'e', 'u'], axis=0, ignore_index=True)
    for i in indexes:
        all_df = [df_all.g_z_kernel[i], df_all.power_s_kernel[i]]
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
            a = dtrange[0: len(df_all.g_z_kernel[i])]
            delta_t.append(a)
        for j in range(0, len(df_all.g_z_kernel[i])):
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
        print(df_all.m[i], df_all.s[i], df_all.e[i], df_all.u[i])
        print(dtrange[dtopt_indx[i]])
        print('dep_list_g_z_index = ',dep_list_index)
        print('inv_list_g_z_index = ',inv_list_index)
        print('selected g_z: ', list_g_z)
        print('selected power: ',list_power)
        print('len selected g_z: ', len(list_g_z))
        print('len selected power: ',len(list_power))
        unit_type = list(df_all.unit_type.apply(lambda x:1 if x == 'excit' else 0))

               
        if unit_type[i] == 1:
            plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=exc_color)
            plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.5)
            plt.plot(1-df_all.power_s_kernel[i][dtopt_indx[i]], df_all.g_z_kernel[i][dtopt_indx[i]], marker="X", markersize=10, markerfacecolor="None", markeredgecolor="red")
            plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
            plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                    alpha=0.5)
            text = []
            for j in range(0, len(list_power)):
                text.append(plt.text(x = 1 - list_power[j], y = list_g_z[j], s = round(delta_t[i][j],3), fontsize = 12))
            adjust_text(text, x=[1 - k for k in list_power], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.ylabel('Pattern ($g_z$)', fontsize = 16)
            plt.xlabel('Residual power', fontsize = 16)
            plt.show()
        else:
            plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=inh_color)
            plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=inh_color, alpha=0.5)
            plt.plot(1-df_all.power_s_kernel[i][dtopt_indx[i]], df_all.g_z_kernel[i][dtopt_indx[i]], marker="X", markersize=10, markerfacecolor="None", markeredgecolor="red")
           
            plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
            plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                    alpha=0.5)
            text = []
            for j in range(0, len(list_power)):
                text.append(plt.text(x = 1 - list_power[j], y = list_g_z[j], s = round(delta_t[i][j],3), fontsize = 12))
            adjust_text(text, x=[1 - k for k in list_power], y= list_g_z, autoalign='xy', only_move={'objects': 'xy', 'points': 'xy', 'text': 'xy'}, force_points=0.0, arrowprops=dict(arrowstyle="->", color='blue', lw=0.7))
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.ylabel('Pattern ($g_z$)', fontsize = 16)
            plt.xlabel('Residual power', fontsize = 16)
            plt.show()

def svd_vs_spatcorr_selected_t_all_units_topt(keys, dtopt_indx, start, stop, ax=None, exc_color='darkorange',
                    inh_color='teal'):
    
    plt.figure(figsize=(12, 8))
    dtrange = (RCParams & keys).fetch1('smoothdts')
    df = pd.DataFrame((RCSVD * RCSpatialAuto.Error * WaveShapeCluster) & keys)
    df = df.sort_values(by=['m', 's', 'e', 'u'])
    df = df.reset_index()
    count_exc = 0
    count_inh = 0
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
        power_opt = []
        g_z_opt = []
        
        try:
            unit_type = list(df.unit_type.apply(lambda x:1 if x == 'excit' else 0))

            for f in dtopt_indx:
                a = dtrange[f-0: f+1]
                delta_t.append(a)
            for j in range(dtopt_indx[i]-0, dtopt_indx[i]+1):
                
                try:
                    if all_df[0][j]>1.96 and all_df[1][j]<=0.95:
                        dep_list_g_z.append(all_df[0][j])
                        dep_list_power.append(all_df[1][j])
                        dep_list_index.append(j)
                        list_g_z.append(all_df[0][j])
                        list_power.append(all_df[1][j])
                        if j == dtopt_indx[i]:
                            power_opt.append(all_df[1][j])
                            g_z_opt.append(all_df[0][j])
                            if unit_type[i] == 1:
                                count_exc += 1
                            else:
                                count_inh += 1
                        else:
                            pass

                    else:
                        inv_list_g_z.append(all_df[0][j])
                        inv_list_power.append(all_df[1][j])
                        inv_list_index.append(j)
                        list_g_z.append(all_df[0][j])
                        list_power.append(all_df[1][j])
                        if j == dtopt_indx[i]:
                            power_opt.append(all_df[1][j])
                            g_z_opt.append(all_df[0][j])
                        else:
                            pass
                except:
                    pass
          
            if unit_type[i] == 1:
                plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=exc_color, alpha=0.7)
                plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=exc_color, alpha=0.3)
                plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
                plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                        alpha=0.5)
                
                plt.ylabel('Pattern (g_z)', fontsize = 18)
                plt.xlabel('Residual power', fontsize = 18)
            else:
                plt.scatter([1 - j for j in inv_list_power], inv_list_g_z, color=inh_color, alpha=0.7)
                plt.scatter([1 - j for j in dep_list_power], dep_list_g_z, color=inh_color, alpha=0.3)
                # plt.plot([1-pow for pow in power_opt], g_z_opt, marker="X", markersize=10,  markerfacecolor="None", markeredgecolor="red")
            
                plt.hlines(1.96, 0, 0.95, linestyles='dashed', alpha=0.5)
                plt.vlines(0.05, 1.96, np.max(all_df[0]), linestyles='dashed',
                        alpha=0.5)
                plt.ylabel('Pattern (g_z)', fontsize = 18)
                plt.xlabel('Residual power', fontsize = 18)
        except:
            pass
    print(count_exc, count_inh)
    plt.xlim(-0.003, 0.78)
    plt.ylim(-3.7, 11.9)
    plt.xticks([0,0.2,0.4,0.6],fontsize = 14)
    plt.yticks([-2,2,6,10],fontsize = 14)
    plt.plot()


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

# example: svd_vs_spatcorr_all_units(df_all, dtopt_indx, 0, 10)

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
