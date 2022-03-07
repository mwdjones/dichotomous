# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:34:25 2022

@author: M. W. Jones
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

'''Import Data'''
folder_path = "/Users/marie/Desktop/Feng Research/FLUX_NET/Bogs and Fens/FLX_CA-SCB_FLUXNET-CH4_2014-2017_1-1/"

scb_daily = pd.read_csv(folder_path + 'FLX_CA-SCB_FLUXNET-CH4_DD_2014-2017_1-1.csv', 
                        parse_dates = ['TIMESTAMP'], 
                        na_values = (-9999))

#%%

'''Sample Plot'''
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(scb_daily.TIMESTAMP, scb_daily.WTD_F, color = 'lightgray')
ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

ax2 = ax1.twinx()
ax2.plot(scb_daily.TIMESTAMP, scb_daily.FCH4_F, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

plt.title('CA - SCB Site, WTD and Methane Flux Dynamics')
plt.savefig(r"C:\Users\marie\Desktop\Feng Research\Figures\Dichotomous Modeling Figures\WTE_dCH4_Fig.pdf")

'''Sample PLot - WTD Flux'''
fig, ax1 = plt.subplots(figsize = (10, 8))
der = np.diff(scb_daily.WTD_F)
der = np.insert(der, 1, np.nan)
ax1.plot(scb_daily.TIMESTAMP, der, color = 'lightgray')
ax1.set(ylabel = 'Water Table Flux (Filled) [m day-1]')

ax2 = ax1.twinx()
ax2.plot(scb_daily.TIMESTAMP, scb_daily.FCH4_F, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

plt.title('CA - SCB Site, WTD Flux and Methane Flux Dynamics')
plt.savefig(r"C:\Users\marie\Desktop\Feng Research\Figures\Dichotomous Modeling Figures\dWTE_dCH4_Fig.pdf")

'''Distributions of WTD and FCH4'''
ax1 = plt.subplot()
plt.scatter(data = scb_daily, x = 'WTD_F', y = 'FCH4_F', c = der, cmap = 'coolwarm', marker = '.')
plt.clim(-0.02, 0.02)
plt.colorbar()
plt.xlabel('Water Table Depth (Filled) [m]')
plt.ylabel('Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

plt.savefig(r"C:\Users\marie\Desktop\Feng Research\Figures\Dichotomous Modeling Figures\WTD_Fig.pdf")

'''Precip and WTD'''
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(scb_daily.TIMESTAMP, scb_daily.WTD_F, color = 'blue')
ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

ax2 = ax1.twinx()
ax2.plot(scb_daily.TIMESTAMP, scb_daily.P_F, color = 'lightgray', zorder = 1)
ax2.set(ylabel = 'Precipitation (Filled) [mm]')

plt.title('CA - SCB Site, Precipitation and WTD')
plt.savefig(r"C:\Users\marie\Desktop\Feng Research\Figures\Dichotomous Modeling Figures\Precip_Fig.pdf")

#%%
from scipy.optimize import curve_fit

'''Parameterization of Precip'''
precip = scb_daily.P_F[scb_daily.P_F > 0].reset_index()

def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c

#Make histogram
counts, bins = np.histogram(precip, density = False)

#Parameterize histogram
popt, pcov = curve_fit(expfunc, xdata = bins, ydata = counts)













