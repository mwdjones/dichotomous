#%%
'''Imports'''
#General python packages
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy import stats
from glob import glob

folder_path = "./data/FLUXNET/"
save_path = "./figures/sites/"

#%%
'''Precip and Air Temp Plot'''
filelist = glob(folder_path + '*/*_DD_*.csv')

atmData = pd.DataFrame({'TIMESTAMP' : [],
                         'TA' : [],
                          'P' : [],
                          'SITE' : []})

for file in filelist:
    sitename = file[-39:-33]
    print(sitename)

    if(sitename == 'US-EDN'): #skip site with less than a year of data
        continue

    data = pd.read_csv(file, 
                       parse_dates = ['TIMESTAMP'],
                       na_values = [-9999])
    temp_dataframe = data[['TIMESTAMP', 'TA', 'P']]
    temp_dataframe['SITE'] = sitename
    temp_dataframe['MON'] = temp_dataframe.TIMESTAMP.dt.month
    temp_dataframe['YEAR'] = temp_dataframe.TIMESTAMP.dt.year

    #for each site create a temperature/precip monthly average plot
    monthly_annualP = temp_dataframe.groupby([temp_dataframe.MON, temp_dataframe.YEAR]).sum().reset_index()
    monthlyP = monthly_annualP.groupby(monthly_annualP.MON).mean().P
    monthlyT = temp_dataframe.groupby(temp_dataframe.TIMESTAMP.dt.month).mean().TA
    #months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    fig, ax = plt.subplots(figsize = (4, 4))
    ax.bar(x = np.arange(1, 13), height = monthlyP, color = 'teal')
    ax.set_ylabel('Total Monthly Precip', color = 'teal')
    ax2 = plt.twinx(ax)
    ax2.plot(monthlyT, 'o-', color = 'darkred', linewidth = 5, markersize = 10)
    ax2.set_ylabel('Average Monthly Temp', color = 'darkred')
    ax.set_xticks([2, 4, 6, 8, 10, 12], ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'])
    ax.set_title(sitename + ' Monthly Climate')
    plt.savefig(save_path + sitename + '_climatedata.jpg', 
        bbox_inches = 'tight')
    plt.show()

    #concatenate
    atmData = pd.concat([atmData, temp_dataframe]).reset_index(drop = True)

# %%

#Same climate plot but all in one big grid
fig, axs = plt.subplots(2, 5, figsize = (20, 6), 
    sharex = True, 
    sharey = True, 
    layout = 'constrained')

sites = ["CA-SCB", "DE-Hte", "DE-SfN", "DE-Zrk", "FI-Si2", "FI-Sii", "FR-LGt", "JP-BBY", "NZ-Kop", "SE-Deg"]

for ax in axs.reshape(10):
    #select data
    if len(sites) == 0:
        continue
    else:
        site = sites[0]
        dat = atmData[atmData.SITE == site]
        del sites[0]

        #repeat the selection
        monthly_annualP = dat.groupby([dat.MON, dat.YEAR]).sum().reset_index()
        monthlyP = monthly_annualP.groupby(monthly_annualP.MON).mean().P
        monthlyT = dat.groupby(dat.TIMESTAMP.dt.month).mean().TA
    

        #plot
        ax.bar(x = np.arange(1, 13), height = monthlyP, color = 'teal')
        ax.set_ylabel('Total Monthly Precip', color = 'teal')
        ax2 = plt.twinx(ax)
        ax2.plot(monthlyT, 'o-', color = 'darkred', linewidth = 5, markersize = 10)
        ax2.set_ylabel('Average Monthly Temp', color = 'darkred')
        ax2.set_ylim(-25, 25)
        ax.set_xticks([2, 4, 6, 8, 10, 12], ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'])
        ax.set_title(site)

plt.savefig(save_path + 'all_climatedata.pdf')
plt.savefig(save_path + 'all_climatedata.svg')
# %%

#Annual Temperature and Precip plot
annualT = atmData.groupby(['SITE', 'YEAR'])['TA'].mean().reset_index()
meanAnnualT = annualT.groupby(['SITE'])['TA'].mean().reset_index()
annualP = atmData.groupby(['SITE', 'YEAR'])['P'].sum().reset_index()
meanAnnualP = annualP.groupby(['SITE'])['P'].mean().reset_index()
annual = pd.merge(meanAnnualP, meanAnnualT, on = 'SITE')

fig, ax = plt.subplots(figsize = (5, 5))

ax.grid(True)
ax.scatter(annual.P, annual.TA, color = 'black', zorder = 2)
ax.set_xlim(0, 1200)
ax.set_ylim(0, 16)
ax.set_xlabel('Total Annual Precipitation [mm]')
ax.set_ylabel('Average Annual Temperature [C]')


#Annotate with site names
for i in range(0, len(annual.SITE)):
    ax.annotate(annual.SITE[i], (annual.P[i] - 100 , annual.TA[i]- 0.75))

plt.savefig(save_path + 'all_climatecomparison.pdf')
plt.savefig(save_path + 'all_climatecomparison.svg')
# %%
