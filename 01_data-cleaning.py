# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:34:25 2022

@author: M. W. Jones

Update 3/30/2022 - Added code section for temperature corrections using the expression from 
the MEF book. Defined a function for deriving the switching rates and the noise time series
from a sample series and a threshold. 

"""
#%%
'''Imports'''
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from glob import glob
import random 
from scipy.optimize import curve_fit
import os
import shutil

random.seed(293)

folder_path = "./data/FLUXNET/"

#%%

filelist = glob(folder_path + '*/*_DD_*.csv')

for file in filelist:
    sitename = file[-39:-33]

    #Import data
    data = pd.read_csv(file, 
                       parse_dates = ['TIMESTAMP'],
                       na_values = [-9999])

    '''Save path for figures'''
    #see if there is a save folder -- if not create one
    save_path = "./figures/data-cleaning/" + sitename + "/"
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)

    #Remove two sites that don't converge when fitting the temperature estimation function
    if((sitename == 'FI-Sii') | (sitename == 'SE-Deg')):
        continue

    '''Sample Timeseries Plot'''
    fig, ax1 = plt.subplots(figsize = (10, 8))
    ax1.plot(data.TIMESTAMP, data.WTD_F, color = 'lightgray')
    ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

    ax2 = ax1.twinx()
    ax2.plot(data.TIMESTAMP, data.FCH4_F, color = 'orange')
    ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

    plt.title(sitename + ' Site, WTD and Methane Flux Dynamics')
    plt.savefig(save_path + "WTE_dCH4_Fig.pdf")
    plt.show()

    '''Sample Plot - WTD Flux'''
    fig, ax1 = plt.subplots(figsize = (10, 8))
    der = np.diff(data.WTD_F)
    der = np.insert(der, 1, np.nan)
    ax1.plot(data.TIMESTAMP, der, color = 'lightgray')
    ax1.set(ylabel = 'Water Table Flux (Filled) [m day-1]')

    ax2 = ax1.twinx()
    ax2.plot(data.TIMESTAMP, data.FCH4_F, color = 'orange')
    ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

    plt.title(sitename + ' Site, WTD Flux and Methane Flux Dynamics')
    plt.savefig(save_path + "dWTE_dCH4_Fig.pdf")
    plt.show()

    '''Distributions of WTD and FCH4'''
    fig, ax1 = plt.subplots(figsize = (10, 8))
    plt.scatter(data = data, x = 'WTD_F', y = 'FCH4_F', c = der, cmap = 'coolwarm', marker = '.')
    plt.clim(-0.02, 0.02)
    plt.colorbar(label = 'Daily Scale Differential')
    plt.xlabel('Water Table Depth (Filled) [m]')
    plt.ylabel('Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

    plt.savefig(save_path + "WTD_Fig.pdf")
    plt.show()

    '''Precip and WTD'''
    fig, ax1 = plt.subplots(figsize = (10, 8))
    ax1.plot(data.TIMESTAMP, data.WTD_F, color = 'blue')
    ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

    ax2 = ax1.twinx()
    ax2.plot(data.TIMESTAMP, data.P_F, color = 'lightgray', zorder = 1)
    ax2.set(ylabel = 'Precipitation (Filled) [mm]')

    plt.title(sitename + ' Site, Precipitation and WTD')
    plt.savefig(save_path + "Precip_Fig.pdf")
    plt.show()

    #################### NORMALIZE CH4 FOR TEMPERATURE #######################

    '''Remove Temperature Dependencies'''
    #Original Plot
    fig, ax1 = plt.subplots(figsize = (6, 6))
    plt.scatter(data.TS_1, data.FCH4_F, alpha = 0.5) #8 cm below the surface
    plt.xlabel("Air Temperature [degC]")
    plt.ylabel("CH4 Flux (Filled) [nmolCH4 m-2 s-1]")

    #Curve fitting must remove nan rows
    data_na = data[["TS_1", "FCH4_F"]].dropna().reset_index(drop = True)
    x = data_na.TS_1
    y = data_na.FCH4_F

    #Fit exponential curve to the data
    popt, pcov = curve_fit(lambda t, a, b: np.exp(-a* (1 / t) + b), x, y, maxfev=50000)
    a = popt[0]
    b = popt[1]

    #Fitted Data
    y_fitted = np.exp(-a* (1 / x) + b)

    #Fitted Plot
    fig, ax1 = plt.subplots(figsize = (6, 6))
    plt.scatter(data_na.TS_1, data_na.FCH4_F, alpha = 0.3)
    plt.scatter(x, y_fitted, color = 'red', marker = '.')
    plt.xlabel("Soil Temperature at 8cm [degC]")
    plt.ylabel("CH4 Flux (Filled) [nmolCH4 m-2 s-1]")

    #Methane Flux controlled for temp
    y_norm = y - y_fitted

    #Fitted Plot w/ Norm
    fig, ax1 = plt.subplots(figsize = (6, 6))
    plt.scatter(x, y, alpha = 0.3, label = 'original')
    plt.scatter(x, y_fitted, color = 'red', marker = '.', label = 'fitted')
    plt.scatter(x, y_norm, color = 'orange', alpha = 0.3, label = 'normalized')
    plt.xlabel("Soil Temperature at 8cm [degC]")
    plt.ylabel("CH4 Flux (Filled) [nmolCH4 m-2 s-1]")
    plt.legend()
    plt.savefig(save_path + "tempFix.pdf")

    '''Apply to Ch4 series'''
    data["FCH4_F_MODEL"] = np.exp(-a* (1 / data.TS_1[data.TS_1 > 0]) + b)
    data["FCH4_F_NORM"] = data.FCH4_F - data.FCH4_F_MODEL

    #Modelled and Actual CH4
    fig, ax1 = plt.subplots(figsize = (10, 8))
    ax1.plot(data.TIMESTAMP, data.FCH4_F, color = 'lightgray')
    ax1.set(ylabel = 'Turbulent Ch4 Flux (Measured, Filled) [m day-1]')

    ax2 = ax1.twinx()
    ax2.plot(data.TIMESTAMP, data.FCH4_F_MODEL, color = 'orange')
    ax2.set(ylabel = 'Turbulent CH4 Flux (Modelled, Filled) [nmolCH4 m-2 s-1]')

    #Norm CH4 and WTE
    fig, ax1 = plt.subplots(figsize = (10, 8))
    der = np.diff(data.WTD_F)
    der = np.insert(der, 1, np.nan)
    ax1.plot(data.TIMESTAMP, der, color = 'lightgray')
    ax1.set(ylabel = 'Water Table Flux (Filled) [m day-1]')

    ax2 = ax1.twinx()
    ax2.plot(data.TIMESTAMP, data.FCH4_F_NORM, color = 'orange')
    ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')
    plt.savefig(save_path + "tempFixWTE.pdf")

    '''Save data'''
    data.to_csv('./data/cleaned/' + sitename + '_normalized.csv')
