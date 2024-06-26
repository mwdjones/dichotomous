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

#%%
'''Setup'''
#Dictionary of site info -- for thresholds and windows of good data
steps = 50
metadata = {'CA-SCB' : {'istart' : 443, 'iend' : 1416, 'thresholds' : np.linspace(-0.35, 0.1, steps)}, 
            'DE-Hte' : {'istart' : 0, 'iend' : 2922, 'thresholds' : np.linspace(-0.6, 0.4, steps)}, 
            'DE-SfN' : {'istart' : 185, 'iend' : 367, 'thresholds' : np.linspace(-0.20, -0.05, steps)}, 
            'DE-Zrk' : {'istart' : 1095, 'iend' : 2190, 'thresholds' : np.linspace(-0.05, 0.75, steps)}, 
            'FI-Si2' : {'istart' : 479, 'iend' : 1018, 'thresholds' : np.linspace(0.06, 0.17, steps)}, 
            'FR-LGt' : {'istart' : 0, 'iend' : 729, 'thresholds' : np.linspace(-0.45, -0.15, steps)}, 
            'JP-BBY' : {'istart' : 0, 'iend' : 1460, 'thresholds' : np.linspace(-0.20, 0.15, steps)}, 
            'NZ-Kop' : {'istart' : 0, 'iend' : 1460, 'thresholds' : np.linspace(-0.25, 0, steps)}, 
            'SE-Deg' : {'istart' : 0, 'iend' : 1825, 'thresholds' : np.linspace(-0.25, 0.3, steps)}}

#%%
'''Run crossing time analysis'''
sites = ["CA-SCB", "DE-Hte", "DE-SfN", "DE-Zrk", "FI-Si2", "FR-LGt", "JP-BBY", "NZ-Kop", "SE-Deg"]
sites_storage = pd.DataFrame({'Site' : np.arange(len(sites)*steps), 
                              'Threshold' : np.arange(len(sites)*steps), 
                              'N_crosses' : np.arange(len(sites)*steps), 
                              'T_avg' : np.arange(len(sites)*steps), 
                              'k1' : np.arange(len(sites)*steps), 
                              'k2' : np.arange(len(sites)*steps), 
                              'CH_avg' : np.arange(len(sites)*steps)})
counter = 0

for site in sites:
    for thresh in metadata[site]['thresholds']:
        daily = pd.read_csv('./data/cleaned/' + site + '_normalized.csv')

        #Step 0: Plot data
        #plt.plot(daily.WTD_F)
        #plt.title('Site: ' + str(site))
        #plt.show()

        #Step 1: Set a single threshold -- will be looped through in future iterations
        WTD_sample = daily.WTD_F[metadata[site]['istart']:metadata[site]['iend']].dropna().reset_index(drop = True)

        #Determine the number of crosses
        n = ncrosses(WTD_sample, thresh)

        #Determine the average time spent above or below before crossing
        t_vec, t = crossTime(WTD_sample, thresh)

        #Correlate this to a k1 and k2 value
        k1param, k2param, _ = switch_param(WTD_sample, thresh)

        #Step 2: Determine average emissions of site
        ch_sample = daily.FCH4_F_NORM[516:1247].dropna().reset_index(drop = True)
        mean_ch = np.mean(ch_sample)

        #Step 3: Log data
        sites_storage['Site'][counter] = site
        sites_storage['Threshold'][counter] = thresh
        sites_storage['N_crosses'][counter] = n
        sites_storage['T_avg'][counter] = t
        sites_storage['k1'][counter] = k1param
        sites_storage['k2'][counter] = k2param
        sites_storage['CH_avg'][counter] = mean_ch
        counter = counter + 1

        #Step 4: Repeat for all sites


# %%
'''Plot'''
fig, ax = plt.subplots(figsize = (5, 5))

ax.grid(True)
ax.scatter(sites_storage.T_avg, sites_storage.CH_avg, color = 'black', zorder = 2)
#ax.set_xlim(0, 180)
ax.set_ylim(-20, 20)
ax.set_xlabel('Average Crossing Time')
ax.set_ylabel('Average Turbulent CH4 Flux [nmolCH4 m-2 s-1]')

# %%
'''Moving Window Version'''
window_length = 30

site_wind = []
wind_no = []
thresh_wind = []
n_wind = []
t_wind = []
k1_wind = []
k2_wind = []
ch_wind = []

for site in sites:
    for thresh in metadata[site]['thresholds']:
        counter = 0
        daily = pd.read_csv('./data/cleaned/' + site + '_normalized.csv')

        #Step 0: Plot data
        #plt.plot(daily.WTD_F)
        #plt.title('Site: ' + str(site))
        #plt.show()

        for i in range(metadata[site]['istart'], metadata[site]['iend']-(window_length-1)):
            #Step 1: Set a single threshold -- will be looped through in future iterations
            #Run a window of specified length
            WTD_sample = daily.WTD_F[i:i+window_length].dropna().reset_index(drop = True)

            #Determine the number of crosses
            n = ncrosses(WTD_sample, thresh)

            #Determine the average time spent above or below before crossing
            t_vec, t = crossTime(WTD_sample, thresh)

            #Correlate this to a k1 and k2 value
            #k1param, k2param, _ = switch_param(WTD_sample, thresh)

            #Step 2: Determine average emissions of site
            ch_sample = daily.WTD_F[i:i+window_length].dropna().reset_index(drop = True)
            mean_ch = np.mean(ch_sample)

            #Step 3: Log data
            site_wind.append(site)
            wind_no.append(counter)
            thresh_wind.append(thresh)
            n_wind.append(n)
            t_wind.append(t)
            #k1_wind.append(k1param)
            #k2_wind.append(k2param)
            ch_wind.append(mean_ch)
            counter = counter + 1

        #Step 4: Repeat for all sites

#%%
#Append lists into DataFrame
sites_storage_window = pd.DataFrame({'Site' : site_wind, 
                              'Window' : wind_no,
                              'Threshold' : thresh_wind, 
                              'N_crosses' : n_wind, 
                              'T_avg' : t_wind, 
                              'CH_avg' : ch_wind})

# %%
'''Plot'''
fig, ax = plt.subplots(figsize = (5, 5))

ax.grid(True)
sns.scatterplot(sites_storage_window.T_avg, sites_storage_window.CH_avg,
                 hue = sites_storage_window.Site, zorder = 2)
#ax.set_xlim(0, 180)
#ax.set_ylim(-20, 20)
ax.set_xlabel('Average Crossing Time')
ax.set_ylabel('Average Turbulent CH4 Flux [nmolCH4 m-2 s-1]')
# %%
