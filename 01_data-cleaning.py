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

'''Import Data'''
folder_path = "data/FLUXNET/CA-SCB/"

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
plt.savefig("figures/data-cleaning/WTE_dCH4_Fig.pdf")
plt.show()

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
plt.savefig("figures/data-cleaning/dWTE_dCH4_Fig.pdf")
plt.show()

'''Distributions of WTD and FCH4'''
fig, ax1 = plt.subplots(figsize = (10, 8))
plt.scatter(data = scb_daily, x = 'WTD_F', y = 'FCH4_F', c = der, cmap = 'coolwarm', marker = '.')
plt.clim(-0.02, 0.02)
plt.colorbar(label = 'Daily Scale Differential')
plt.xlabel('Water Table Depth (Filled) [m]')
plt.ylabel('Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

plt.savefig("figures/data-cleaning/WTD_Fig.pdf")
plt.show()

'''Precip and WTD'''
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(scb_daily.TIMESTAMP, scb_daily.WTD_F, color = 'blue')
ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

ax2 = ax1.twinx()
ax2.plot(scb_daily.TIMESTAMP, scb_daily.P_F, color = 'lightgray', zorder = 1)
ax2.set(ylabel = 'Precipitation (Filled) [mm]')

plt.title('CA - SCB Site, Precipitation and WTD')
plt.savefig("figures/data-cleaning/Precip_Fig.pdf")
plt.show()

#%%

##########################################################################
#################### NORMALIZE CH4 FOR TEMPERATURE #######################
##########################################################################

'''Remove Temperature Dependencies'''
import random 
from scipy.optimize import curve_fit
random.seed(293)

#Original Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(scb_daily.TS_4, scb_daily.FCH4_F, alpha = 0.5) #8 cm below the surface
plt.xlabel("Air Temperature [degC]")
plt.ylabel("CH4 Flux (Filled) [nmolCH4 m-2 s-1]")

#Curve fitting must remove nan rows
scb_daily_na = scb_daily[["TS_4", "FCH4_F"]].dropna().reset_index(drop = True)
x = scb_daily_na.TS_4
y = scb_daily_na.FCH4_F

#Fit exponential curve to the data
popt, pcov = curve_fit(lambda t, a, b: np.exp(-a* (1 / t) + b), x, y)
a = popt[0]
b = popt[1]

#Fitted Data
y_fitted = np.exp(-a* (1 / x) + b)

#Fitted Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(scb_daily_na.TS_4, scb_daily_na.FCH4_F, alpha = 0.3)
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
plt.savefig("figures/data-cleaning/tempFix.pdf")

#%%
'''Apply to Ch4 series'''

scb_daily["FCH4_F_MODEL"] = np.exp(-a* (1 / scb_daily.TS_4[scb_daily.TS_4 > 0]) + b)
scb_daily["FCH4_F_NORM"] = scb_daily.FCH4_F - scb_daily.FCH4_F_MODEL

#Modelled and Actual CH4
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(scb_daily.TIMESTAMP, scb_daily.FCH4_F, color = 'lightgray')
ax1.set(ylabel = 'Turbulent Ch4 Flux (Measured, Filled) [m day-1]')

ax2 = ax1.twinx()
ax2.plot(scb_daily.TIMESTAMP, scb_daily.FCH4_F_MODEL, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Modelled, Filled) [nmolCH4 m-2 s-1]')

#Norm CH4 and WTE
fig, ax1 = plt.subplots(figsize = (10, 8))
der = np.diff(scb_daily.WTD_F)
der = np.insert(der, 1, np.nan)
ax1.plot(scb_daily.TIMESTAMP, der, color = 'lightgray')
ax1.set(ylabel = 'Water Table Flux (Filled) [m day-1]')

ax2 = ax1.twinx()
ax2.plot(scb_daily.TIMESTAMP, scb_daily.FCH4_F_NORM, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')
plt.savefig("figures/data-cleaning/tempFixWTE.pdf")

#Norm CH4 and WTD
ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(data = scb_daily, x = 'WTD_F', y = 'FCH4_F_NORM', alpha = 0.5)
plt.xlabel('Water Table Depth (Filled) [m]')
plt.ylabel('Normalized Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

#%%
'''Save data'''

scb_daily.to_csv('data/cleaned/CA-SCB/CA-SCB_normalized.csv')

#%%
##########################################################################
#################### PARAMETRIZE WATER TABLE DEPTH #######################
##########################################################################

#Water Table Time series
fig, ax = plt.subplots(figsize = (8, 6))
plt.plot(scb_daily.TIMESTAMP, scb_daily.WTD_F)
ax.set(ylabel = 'Water Table Depth (Filled) [m]')

#Potential fluctuation thresholds
thresh = [0.05, 0, -0.05, -0.1, -0.15, -0.2, -0.25]

#Select a bit of WTE data that contains no missing data. 
WTD_sample = scb_daily.WTD_F[516:1247].dropna()

fig, ax = plt.subplots(figsize = (8, 5))
plt.plot(WTD_sample, zorder = 5, color = 'lightgray')
#for i in thresh:
#    plt.axhline(y = i, color = 'r', linestyle = '--', alpha = 0.5)
ax.set(ylabel = 'Water Table Depth (Filled) [m]')
plt.savefig("figures/data-cleaning/thresholds.pdf")


'''Run on the water table data'''
storage_N = pd.DataFrame(np.zeros((len(WTD_sample), len(thresh))))

for i in range(0, len(thresh)):
    #Threshold
    t = thresh[i]
    
    #Calculate values
    k1_p, k2_p, N = switch_param(WTD_sample, t)
    
    #Save switching rates
    
    
    #Save derived noise
    storage_N[i] = N
    
#%%
'''Plot the derived noise time series'''

fig, ax = plt.subplots(len(thresh), 1, figsize=(8, 5))
    
for i in range(0, len(thresh)):
    ax[i].step(x = storage_N.index, y = storage_N[i])
    ax[i].set(xlim = (0, len(WTD_sample)))
    ax[i].set(ylim = (- 1.25, 1.25))
    ax[i].set(xlabel = 'Time [Day]') 
    ax[i].text(0, -1.75, str(thresh[i]) + "m", fontsize = 8, color = 'black')
    
    #Aesthetic
    ax[i].yaxis.set_visible(False)
    
    if i == len(thresh) - 1:
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)       
        ax[i].spines["bottom"].set_position(("axes", -0.3))
    else:
        ax[i].set_frame_on(False)
        ax[i].xaxis.set_visible(False)

plt.savefig("figures/data-cleaning/derivednoise2.pdf")



# %%
