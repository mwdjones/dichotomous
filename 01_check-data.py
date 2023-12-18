#%%

'''Imports'''
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from glob import glob

folder_path = "./data/FLUXNET/"

# %%

'''Import Files'''
filelist = glob(folder_path + '*/*_DD_*.csv')

for file in filelist:
    sitename = file[-39:-33]
    print(sitename)

    data = pd.read_csv(file, 
                       parse_dates = ['TIMESTAMP'],
                       na_values = [-9999])

    print('Temperature variables: ' + str([col for col in data.columns if 'TS' in col]))

    if('WTD_F' in data.columns):
        #Plot the water table and CH4 flux
        fig, ax1 = plt.subplots(figsize = (6, 4))
        ax1.plot(data.TIMESTAMP, data.WTD_F, color = 'blue')
        ax1.set_ylabel('Water Table Depth (Filled) [m]', color = 'blue')
        ax1.set_ylim(-1,1)

        ax2 = ax1.twinx()
        ax2.plot(data.TIMESTAMP, data.FCH4_F, color = 'orange')
        ax2.set_ylabel('Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]', color = 'orange')

        plt.title(sitename + ' Site, WTD and Methane Flux Dynamics')
        plt.savefig('./figures/timeseries/plottedData_' + sitename + '.jpg')
        plt.show()
    else:
        print('Site does not contain WTD data.')

# %%
