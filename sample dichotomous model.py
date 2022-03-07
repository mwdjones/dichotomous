# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:22:47 2022

@author: M. W. Jones. 

Updated 3/3/2022 - Function script written to generate dichotomous noise (generate_noise). Generated 
solutions for the basic methane case using both numerical and analytical methods. Analytical solution looks as expected
numerical solution is off (solved). General solution for methane pdf generated using the steady state pdf.
"""
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
'''Sample Dichotomous System'''

#dC/dt = \theta(1-C), when y > yc (inundated state); -C, when c < yc (aerated state)
#State 1 --> Inundated
#State 0 --> Aerated

'''Parameters'''
#General
k1 = 0.7 #switching rate from state 1 to state 0
k2 = 0.7 #switching rate from state 0 to state 1
time = 30 #length of time
dt = 0.1 #time step 

#Data params
k_p = 4
k_ox = 3
k_om = 10
k_e = 7

k_max = 20
a_max = 1000

#Nondimensionalized params
#DO NOT EDIT
b = k_p/k_ox
g = k_p/k_om
e = k_e/k_max
a = 1/a_max

#%%
'''Generate Noise Using EXP'''
def generate_noise(k1, k2, num, step):
    #Takes in switching rates and generates symmetric dichotomous noise for a specified time period and step. 
    #Inputs: k1, k2 - The switching rates from state 1 to 0 and 0 to 1 respectively
    #        num - length of time 
    #        step - time step 
    #Outputs: A pandas dataframe containing two columns, one with the time and one with the noise value (either -1 or 1)
    
    #Initialize the noise
    switch = []

    #Generate random switching
    for i in range(0, num+10): #buffer
        if i%2 == 0:
            switch.append(-np.ones(int(np.random.exponential(1/k2, 1)/step)))
        else:
            switch.append(np.ones(int(np.random.exponential(1/k1, 1)/step)))
            
    #Merge frames into vector
    switch = np.concatenate(switch)
    switch = switch[0 : int(num/step)]
    
    #Generate x values
    x = np.arange(0, num, step)

    return pd.DataFrame({'x' : x,'noise' : switch})

#Generate Noise
noise = generate_noise(k1, k2, time, dt)

#Plot
plt.step(noise.x, noise.noise)
plt.xlim(0, time)
plt.xlabel('Time [t]')
plt.ylabel(r'$\xi(t)$') 

#%%
##############################################################################
###################### ANALYTICAL SOLUTION ###################################
##############################################################################

'''Exponential Decay - Solving the Dynamic Equation'''
#phi(t) decreases exponentially in state 0 and increases exponentially in state 1
#i.e. phi1(t) = 1 - phi and phi2(t) = -phi

### Euler Method - Solves for phi and dphi/dt using the step specified
noise['dphi'] = np.zeros(len(noise.noise))
noise['phi'] = np.zeros(len(noise.noise))

#Initial Concentration
noise.phi[0] = 0.5

for i in range(1, len(noise.phi)):
    if(noise.noise[i] == 1):
        #State 1
        noise.dphi[i] = b*(1 - noise.phi[i-1])*dt
        noise.phi[i] = noise.dphi[i] + noise.phi[i-1]
    else:
        #State 2
        noise.dphi[i] = (-noise.phi[i-1])*dt
        noise.phi[i] = noise.dphi[i] + noise.phi[i-1]

### Analytical Solution 

noise['f_phi'] = np.zeros(len(noise.noise))
noise['g_phi'] = np.zeros(len(noise.noise))
noise['dphi_a'] = np.zeros(len(noise.noise))
noise['phi_a'] = np.zeros(len(noise.noise))

#Initial Concentration
noise.phi_a[0] = 0.5

for i in range(1, len(noise.phi)):
    noise.f_phi[i] = 0.5*(b*(1 - noise.phi[i-1]) - noise.phi[i-1])
    noise.g_phi[i] = 0.5*(b*(1 - noise.phi[i-1]) + noise.phi[i-1])
    noise.dphi_a[i] = (noise.f_phi[i] + noise.g_phi[i]*noise.noise[i])*dt
    noise.phi_a[i] = noise.phi_a[i-1] + noise.dphi_a[i]

#%%
'''Plot Panel'''
fig, ax = plt.subplots(3, 1, figsize=(7, 10))

ax[0].step(noise.x, noise.noise)
ax[0].set(xlim = (0, time))
ax[0].set(xlabel = 'Time [t]', ylabel = r'$\xi(t)$') 

ax[1].plot(noise.x, noise.phi, '--', label = 'Euler')
ax[1].plot(noise.x, noise.phi_a, color = 'orange', zorder = 0, label = 'Analytical')
ax[1].set(xlim = (0, time))
ax[1].set(xlabel = 'Time [t]', ylabel = r'$\phi$')
ax[1].legend(loc = 'upper right')

ax[2].plot(noise.x, noise.dphi, '--', label = 'Euler')
ax[2].plot(noise.x, noise.dphi_a, color = 'orange', zorder = 0, label = 'Analytical')
ax[2].set(ylim = (-0.2, 0.2), xlim = (0, time))
ax[2].set(xlabel = 'Time [t]', ylabel = r'$\frac{d\phi}{dt}$')
ax[2].legend(loc = 'upper right')

plt.savefig(r"C:\Users\marie\Desktop\Feng Research\Figures\Dichotomous Modeling Figures\simplesystem.pdf")

#%%
'''PDF of phi'''

### Empirical pdf
fig, ax = plt.subplots()
n, bins, patches = ax.hist(noise.phi, 50, density = True)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel('Probability density (Not really)')
plt.xlim(0, 1)
plt.show()

### Empirical cdf
fig, ax = plt.subplots()
n, bins, patches = ax.hist(noise.phi, 50, density = True, histtype = 'step', cumulative = True)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel('Cumulative density')
plt.xlim(0, 1)
plt.show()

### Theoretical pdf
#Range of phi values
pdf_phi = np.arange(0.01, 1, step = 0.01)

term1 = [math.pow(value, k2) for value in pdf_phi]
term2 = [math.pow(1 - value, k1/b) for value in pdf_phi]
pdf_un = b*((1/(b*(1 - pdf_phi))) + (1/(pdf_phi)))*term1*term2
norm_constant = sum(pdf_un)
pdf = pdf_un/norm_constant

#Plot
plt.plot(pdf_phi, pdf)
plt.xlim(0, 1)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$P[\phi]$')
plt.show()

#%%
'''Range of k1, k2 values'''

#Generate k values
k1_samp = np.arange(0.2, 3, step = 0.4)
k2_samp = np.arange(3, 0.2, step = -0.4)

#Zip to list
values = list(zip(k1_samp, k2_samp))

#Generate Keys
keys = np.arange(0, 7, step = 1)

#Zip to dict
k_vals = dict(zip(keys, values)) 

for key in k_vals:
    k1 = k_vals[key][0]
    k2 = k_vals[key][1]
    
    term1 = [math.pow(value, k2) for value in pdf_phi]
    term2 = [math.pow(1 - value, k1/b) for value in pdf_phi]
    pdf_un = b*((1/(b*(1 - pdf_phi))) + (1/(pdf_phi)))*term1*term2
    norm_constant = sum(pdf_un)
    pdf = pdf_un/norm_constant
    
    plt.plot(pdf_phi, pdf, label = r'$k_1$ = ' + str(round(k1, 2)) + r', $k_2$ = ' + str(round(k2, 2)))
    plt.xlim(0, 1)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$P[\phi]$')
    plt.legend(loc = 'upper center')
    
plt.savefig(r"C:\Users\marie\Desktop\Feng Research\Figures\Dichotomous Modeling Figures\betapdf.pdf")
    
#%%










