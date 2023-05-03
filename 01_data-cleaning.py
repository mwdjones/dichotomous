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

#%%
'''CA-SCB'''

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
'''JP-BBY'''
'''Import Data'''
folder_path = "data/FLUXNET/JP-BBY/"

bby_daily = pd.read_csv(folder_path + 'FLX_JP-BBY_FLUXNET-CH4_DD_2015-2018_1-1.csv', 
                        parse_dates = ['TIMESTAMP'], 
                        na_values = (-9999))

'''Sample Plot'''
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(bby_daily.TIMESTAMP, bby_daily.WTD_F, color = 'lightgray')
ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

ax2 = ax1.twinx()
ax2.plot(bby_daily.TIMESTAMP, bby_daily.FCH4_F, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

#Original Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(bby_daily.TS_4, bby_daily.FCH4_F, alpha = 0.5) #8 cm below the surface
plt.xlabel("Air Temperature [degC]")
plt.ylabel("CH4 Flux (Filled) [nmolCH4 m-2 s-1]")

#Curve fitting must remove nan rows
bby_daily_na = bby_daily[["TS_4", "FCH4_F"]].dropna().reset_index(drop = True)
x = bby_daily_na.TS_4
y = bby_daily_na.FCH4_F

#Fit exponential curve to the data
popt, pcov = curve_fit(lambda t, a, b: np.exp(-a* (1 / t) + b), x, y)
a = popt[0]
b = popt[1]

#Fitted Data
y_fitted = np.exp(-a* (1 / x) + b)

#Fitted Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(bby_daily_na.TS_4, bby_daily_na.FCH4_F, alpha = 0.3)
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

bby_daily["FCH4_F_MODEL"] = np.exp(-a* (1 / bby_daily.TS_4[bby_daily.TS_4 > 0]) + b)
bby_daily["FCH4_F_NORM"] = bby_daily.FCH4_F - bby_daily.FCH4_F_MODEL

#Modelled and Actual CH4
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(bby_daily.TIMESTAMP, bby_daily.FCH4_F, color = 'lightgray')
ax1.set(ylabel = 'Turbulent Ch4 Flux (Measured, Filled) [m day-1]')

ax2 = ax1.twinx()
ax2.plot(bby_daily.TIMESTAMP, bby_daily.FCH4_F_MODEL, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Modelled, Filled) [nmolCH4 m-2 s-1]')

#Save
bby_daily.to_csv('data/cleaned/JP-BBY_normalized.csv')
# %%
'''NZ-KOP'''
'''Import Data'''
folder_path = "data/FLUXNET/NZ-KOP/"

kop_daily = pd.read_csv(folder_path + 'FLX_NZ-KOP_FLUXNET-CH4_DD_2012-2015_1-1.csv', 
                        parse_dates = ['TIMESTAMP'], 
                        na_values = (-9999))

'''Sample Plot'''
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(kop_daily.TIMESTAMP, kop_daily.WTD_F, color = 'lightgray')
ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

ax2 = ax1.twinx()
ax2.plot(kop_daily.TIMESTAMP, kop_daily.FCH4_F, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

#Original Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(kop_daily.TS_3, kop_daily.FCH4_F, alpha = 0.5) #8 cm below the surface
plt.xlabel("Air Temperature [degC]")
plt.ylabel("CH4 Flux (Filled) [nmolCH4 m-2 s-1]")

#Curve fitting must remove nan rows
kop_daily_na = kop_daily[["TS_3", "FCH4_F"]].dropna().reset_index(drop = True)
x = kop_daily_na.TS_3
y = kop_daily_na.FCH4_F

#Fit exponential curve to the data
popt, pcov = curve_fit(lambda t, a, b: np.exp(-a* (1 / t) + b), x, y)
a = popt[0]
b = popt[1]

#Fitted Data
y_fitted = np.exp(-a* (1 / x) + b)

#Fitted Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(kop_daily_na.TS_3, kop_daily_na.FCH4_F, alpha = 0.3)
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

kop_daily["FCH4_F_MODEL"] = np.exp(-a* (1 / kop_daily.TS_3[kop_daily.TS_3 > 0]) + b)
kop_daily["FCH4_F_NORM"] = kop_daily.FCH4_F - kop_daily.FCH4_F_MODEL

#Modelled and Actual CH4
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(kop_daily.TIMESTAMP, kop_daily.FCH4_F, color = 'lightgray')
ax1.set(ylabel = 'Turbulent Ch4 Flux (Measured, Filled) [m day-1]')

ax2 = ax1.twinx()
ax2.plot(kop_daily.TIMESTAMP, kop_daily.FCH4_F_MODEL, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Modelled, Filled) [nmolCH4 m-2 s-1]')

#Save
kop_daily.to_csv('data/cleaned/NZ-KOP_normalized.csv')
# %%
'''SE-DEG'''
'''Import Data'''
folder_path = "data/FLUXNET/SE-DEG/"

deg_daily = pd.read_csv(folder_path + 'FLX_SE-DEG_FLUXNET-CH4_DD_2014-2018_1-1.csv', 
                        parse_dates = ['TIMESTAMP'], 
                        na_values = (-9999))

'''Sample Plot'''
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(deg_daily.TIMESTAMP, deg_daily.WTD_F, color = 'lightgray')
ax1.set(ylabel = 'Water Table Depth (Filled) [m]')

ax2 = ax1.twinx()
ax2.plot(deg_daily.TIMESTAMP, deg_daily.FCH4_F, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

#Original Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(deg_daily.TS_4, deg_daily.FCH4_F, alpha = 0.5) #8 cm below the surface
plt.xlabel("Air Temperature [degC]")
plt.ylabel("CH4 Flux (Filled) [nmolCH4 m-2 s-1]")

#Curve fitting must remove nan rows
deg_daily_na = deg_daily[["TS_4", "FCH4_F"]].dropna().reset_index(drop = True)
x = deg_daily_na.TS_4
y = deg_daily_na.FCH4_F

#Fit exponential curve to the data
popt, pcov = curve_fit(lambda t, a, b: np.exp(-a* (1 / t) + b), x, y)
a = popt[0]
b = popt[1]

#Fitted Data
y_fitted = np.exp(-a* (1 / x) + b)

#Fitted Plot
fig, ax1 = plt.subplots(figsize = (6, 6))
plt.scatter(deg_daily_na.TS_4, deg_daily_na.FCH4_F, alpha = 0.3)
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

deg_daily["FCH4_F_MODEL"] = np.exp(-a* (1 / deg_daily.TS_4[deg_daily.TS_4 > 0]) + b)
deg_daily["FCH4_F_NORM"] = deg_daily.FCH4_F - deg_daily.FCH4_F_MODEL

#Modelled and Actual CH4
fig, ax1 = plt.subplots(figsize = (10, 8))
ax1.plot(deg_daily.TIMESTAMP, deg_daily.FCH4_F, color = 'lightgray')
ax1.set(ylabel = 'Turbulent Ch4 Flux (Measured, Filled) [m day-1]')

ax2 = ax1.twinx()
ax2.plot(deg_daily.TIMESTAMP, deg_daily.FCH4_F_MODEL, color = 'orange')
ax2.set(ylabel = 'Turbulent CH4 Flux (Modelled, Filled) [nmolCH4 m-2 s-1]')

#Save
deg_daily.to_csv('data/cleaned/SE-DEG_normalized.csv')
# %%
