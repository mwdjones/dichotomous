#%%

'''Imports'''
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from glob import glob

folder_path = "D:/1_DesktopBackup/Feng Research/Data/FLUX_NET/Bogs and Fens/"

# %%

'''Import Files'''
filelist = glob(folder_path + 'FLX_*/*/*_DD_*.csv')

for file in filelist:
    data = pd.read_csv(file, 
                       na_values = [-9999])

    #Plot the water table and CH4 flux
    fig, ax1 = plt.subplots(figsize = (10, 8))
    ax1.plot(data.TIMESTAMP, data.WTD_F, color = 'lightgray')
    ax1.set(ylabel = 'Water Table Depth (Filled) [m]')
    ax1.set_ylim(-1,1)

    #ax2 = ax1.twinx()
    #ax2.plot(data.TIMESTAMP, data.FCH4_F, color = 'orange')
    #ax2.set(ylabel = 'Turbulent CH4 Flux (Filled) [nmolCH4 m-2 s-1]')

    plt.title(file[-39:-33] + ' Site, WTD and Methane Flux Dynamics')
    plt.show()
# %%
