#%%
'''
WORKFLOW
Cleaning
1. Import site data
2. Take years that have beengapfilled, remove NA
3. Plot to check

Analysis
4. Select threshold limits and generate values
5. Get noise and switching rates at each threshold
6. Run methane model on noise
7. Store and group by threshld to get pdf

Export
8. Plot and save
'''

'''Imports'''
#General python packages
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy import stats

#For solving systems of equations
import sympy as sp
from sympy.solvers import solve

#Utils
from functions import *

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
'''(1) Import Normalized Data'''
#site list: "CA-SCB", "DE-Hte", "DE-SfN", "DE-Zrk", "FI-Si2", "FI-Sii", "FR-LGt", "JP-BBY", "NZ-KOP", "SE-DEG"
sitename = "CA-SCB"
daily = pd.read_csv('data/cleaned/' + sitename + '_normalized.csv')

#%%
'''(2, 3) Select Data'''
fig, ax = plt.subplots(figsize = (8, 5))
plt.plot(daily.WTD_F, zorder = 5, color = 'lightgray')
ax.set(ylabel = 'Water Table Depth (Filled) [m]')
plt.suptitle(sitename)

#%%
#Select a bit of WTE data that contains no missing data. 
#   CA-SCB [516:1247]
#   JP-BBY [all]
#   NZ-NOP [all]
#   SE-DEG
WTD_sample = daily.WTD_F[516:1247].dropna()
meth_sample = daily.FCH4_F_NORM[516:1247]

fig, ax = plt.subplots(figsize = (8, 5))
plt.plot(WTD_sample, zorder = 5, color = 'lightgray')
ax.set(ylabel = 'Water Table Depth (Filled) [m]')
plt.suptitle(sitename)

#%%
'''(4, 5) Pull switching rates from each of the chosen thresholds'''
nbin = 12
l_limit = -0.25 
u_limit = 0.05
#Redefine thresh to increase samples
#   CA-SCB np.linspace(-0.25, 0.05, 30)
#   JP-BBY np.linspace(-0.20, 0.15, 30)
#   NZ-KOP np.linspace(-0.25, -0.05, 30)
thresh = np.linspace(l_limit, u_limit, nbin)

#Rerun decomp
storage_N = pd.DataFrame(np.zeros((len(WTD_sample), len(thresh))))

#The probability of being in state 1 is k2/(k1+k2) and in state 2 is k1/(k1+k2)
k1_values = []
k2_values = []
crosses = []

for i in range(0, len(thresh)):
    #Threshold
    t = thresh[i]

    #Derive noise
    k1, k2, N = switch_param(WTD_sample, t)

    #Save derived noise
    storage_N[i] = N

    #Calculate crosses
    cross = ncrosses(WTD_sample.reset_index(drop=True), t)

    k1_values.append(k1)
    k2_values.append(k2)
    crosses.append(cross)

#Combine into a dataframe
derivedK = pd.DataFrame([thresh, k1_values, k2_values, crosses]).transpose()
derivedK.columns = ['thresholds', 'k1', 'k2', 'crosses']

'''(6) Run coupled methane/acetate simulation'''
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

'''(6.5) Linear composition of methane emissions to simulate total emissions signature across all thresholds'''

#reshape 
storage_thresh_reshaped = pd.pivot(columns = 'Threshold') 

storage_thresh_reshaped['Accumulated'] = storage_thresh_reshaped.sum(axis = 1, skipna = True)

#Plot simulated methane emission trace against measured methane trace
fig, ax = plt.subplots(1, 1, figsize=(5, 3))

plt.plot(meth_sample, color = 'lightgray')
ax.set_ylabel("Turbulent Ch4 Flux (Measured, Filled) [m day-1]")

ax2 = ax.twinx()
plt.plot(storage_thresh_reshaped.Accumulated, color = 'blue')
ax2.set_ylabel('Estimated Cumulative $\frac{dE}{dt}$', color = 'blue')
ax.set_xlabel('Time')

plt.savefig("figures/sites/" + sitename + "_methanecomparison.pdf", bbox_inches = 'tight')
plt.show() 

'''(7) Groupby threshold to get average emission'''
thresh_pdf = storage_thresh.groupby('Threshold').mean()

#%%
'''(8) Plot'''
fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.plot(thresh_pdf.index, thresh_pdf.Emission, 
           marker = 'o')

ax.set_xlim(l_limit, u_limit)
ax.set_xlabel('Threshold')
ax.set_ylabel(r'Average $\frac{dE}{dt}$')

plt.savefig("figures/sites/" + sitename + "_emissions.pdf", bbox_inches = 'tight')
plt.show() 

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.scatter(x = derivedK.crosses, y = thresh_pdf.Emission, 
           s = thresh_pdf.Emission*80,
           alpha = 0.7,
           marker = 'o')

#ax.set_xlim(-0.25, 0.05)
ax.set_xlabel('Times WTE crosses the threshold')
ax.set_ylabel(r'Average $\frac{dE}{dt}$')

plt.savefig("figures/sites/" + sitename + "_counts.pdf", bbox_inches = 'tight')
plt.show() 

# %%

'''(9) Plot - binned model/data comparison'''
bin_means, bin_edges, binnumber = stats.binned_statistic(WTD_sample, meth_sample, statistic = np.nanmean, bins = nbin, range = [l_limit, u_limit])

#Janky ass plot - adjust later
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
sns.stripplot(binnumber, meth_sample, alpha = 0.1, color = 'blue', ax = ax)
ax.plot(bin_means, color='blue', lw=2, marker = 'o', label = 'Raw Data')
ax.set_ylabel("Turbulent Ch4 Flux (Measured, Filled) [m day-1]", color = 'blue')
ax.set_xlabel("Threshold number/WTD bin")

ax2 = ax.twinx()
ax2.plot(range(0, len(thresh)), 10*thresh_pdf.Emission, marker = 'o', color = 'red', label = 'Model Data')
ax2.set_ylabel('Average dE/day', color = 'red')

#plt.legend()

plt.savefig("figures/sites/" + sitename + "_jankysample.pdf", bbox_inches = 'tight')
# %%
