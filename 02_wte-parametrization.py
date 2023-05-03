#%%

'''Imports'''
#General python packages
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

#For solving systems of equations
import sympy as sp
from sympy.solvers import solve

#Utils
from functions import *

'''Import Normalized Data'''
scb_daily = pd.read_csv('data/cleaned/CA-SCB/CA-SCB_normalized.csv')

#%%
'''Params'''
#Params
k_p = 2
k_ox = 1
k_om = 6
k_e = 0.2

k_max = 1
a_max = 1000

#Nondimensionalized
b = k_p/k_ox
g = k_p/k_om
e = k_e/k_max
a = 1/a_max

dt = 1

#%%
##########################################################################
#################### PARAMETRIZE WATER TABLE DEPTH #######################
##########################################################################

#Potential fluctuation thresholds
thresh = [0.05, 0, -0.05, -0.1, -0.15, -0.2, -0.25]

#Select a bit of WTE data that contains no missing data. 
WTD_sample = scb_daily.WTD_F[516:1247].dropna()

fig, ax = plt.subplots(figsize = (8, 5))
plt.plot(WTD_sample, zorder = 5, color = 'lightgray')
for i in thresh:
    plt.axhline(y = i, color = 'r', linestyle = '--', alpha = 0.5)
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

#%%
'''Plot the modelled methane from the derived noise time series'''

fig, ax = plt.subplots(len(thresh), 1, figsize=(8, 5))

for i in range(0, len(thresh)):
    #take noise data
    derivedNoise = pd.DataFrame({'x' : storage_N.index,
                         'noise' : storage_N[i]})
    derivedNoise_resampled = downsample(derivedNoise, 'x', 0.1)

    #calculate methane
    derivedMethane = rk4_solve_ma(0.5, 0.5, 0.1, derivedNoise_resampled, kp = k_p, kox = k_ox, kom = k_om, e = e)
    ax[i].step(x = storage_N.index, y = derivedNoise.noise)
    ax[i].plot(derivedMethane.Time, derivedMethane.Emission)
    ax[i].set(xlim = (0, len(WTD_sample)))
    #ax[i].set(ylim = (- 1.25, 1.25))
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
'''Pull switching rates from each of the chose thresholds'''
#Redefine thresh to increase samples
#thresh = np.linspace(-0.25, 0.05, 30)

#Rerun decomp
storage_N = pd.DataFrame(np.zeros((len(WTD_sample), len(thresh))))

#The probability of being in state 1 is k2/(k1+k2) and in state 2 is k1/(k1+k2)
k1_values = []
k2_values = []

for i in range(0, len(thresh)):
    #Threshold
    t = thresh[i]

    #Derive noise
    k1, k2, N = switch_param(WTD_sample, t)

    #Save derived noise
    storage_N[i] = N

    #Calculate switching
    #k1, k2 = calc_rates(N)

    k1_values.append(k1)
    k2_values.append(k2)

#Combine into a dataframe
derivedK = pd.DataFrame([thresh, k1_values, k2_values]).transpose()
derivedK.columns = ['thresholds', 'k1', 'k2']

# %%
'''Run coupled methane/acetate simulation'''
#Use the switching rates previously derived

#Empty df for values
storage_thresh = pd.DataFrame({'Time' : [], 
                               'Noise' : [], 
                               'Acetate': [], 
                               'Methane' : [], 
                               'Emission' : [], 
                               'Threshold' : []})

for i in range(0, len(thresh)):
    noise = pd.DataFrame({'x' : np.array(range(0, len(WTD_sample))),
                         'noise' : np.array(storage_N[i])})
    
    noise_resampled = downsample(noise, 'x', 0.1)

    #Run numerical solution
    s = rk4_solve_ma(0.5, 0.5, 0.1, noise_resampled,
                     kp = k_p, kox = k_ox, kom = k_om, e = e)
    
    s['Threshold'] = thresh[i]

    #Concat
    storage_thresh = pd.concat((storage_thresh, s))


# %%

'''Groupby threshold to get average emission'''

thresh_pdf = storage_thresh.groupby('Threshold').mean()

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.plot(thresh_pdf.index, thresh_pdf.Emission, 
           marker = 'o')

ax.set_xlim(-0.25, 0.05)
ax.set_xlabel('Threshold')
ax.set_ylabel(r'Average $\frac{dE}{dt}$')


plt.show()


# %%
