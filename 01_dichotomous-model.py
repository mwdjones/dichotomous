# -*- coding: utf-8 -*-
#%%
"""
Created on Thu Feb 17 16:22:47 2022

@author: M. W. Jones. 

Updated 3/3/2022 - Function script written to generate dichotomous noise (generate_noise). Generated 
solutions for the basic methane case using both numerical and analytical methods. Analytical solution looks as expected
numerical solution is off (solved). General solution for methane pdf generated using the steady state pdf.

Updated 3/14/2022 - Run code for b = 4, 1, 0.25 and stored in the triptych.pdf file in dichotomous figures

Updated 3/15/2022 - Placed analytical solver and plotting code in functions for easier application.

Update 3/16/2022 - Added the analytical solution for the pdf of phi as a function, generate pdf. 

Update 3/18/2022 - Added numerical solution for the acetate and methane equations and adjusted the equations as recorded in the slides. 
Still unsure if these are the right equations to use -- will consult with Xue on Tuesday. 

Update 3/23/2022 - Adjustments made to the equations so they are now correct and include the solution for emissions. They are currently in the 
dimensional form. Numerical analysis added to compute the emissions for an array of possible k1 and k2 values.

"""

#%%
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import *

#%%
'''Parameters'''
#dC/dt = \theta(1-C), when y > yc (inundated state); -C, when c < yc (aerated state)
#State 1 --> Inundated
#State 0 --> Aerated

#General
k1 = 1 #switching rate from state 1 to state 0
k2 = 1 #switching rate from state 0 to state 1
time = 30 #length of time
dt = 0.1 #time step 

#Data params
k_p = 2
k_ox = 1
k_om = 6
k_e = 0.2

k_max = 1
a_max = 1000

#Nondimensionalized params
#DO NOT EDIT
b = k_p/k_ox
g = k_p/k_om
e = k_e/k_max
a = 1/a_max

#%%
'''Generate Noise Using EXP'''
#Generate Noise
noise = generate_noise(k1, k2, time, dt)

#Plot
plt.step(noise.x, noise.noise)
plt.xlim(0, time)
plt.xlabel('Time [t]')
plt.ylabel(r'$\xi(t)$') 

#%%
##############################################################################
######### ANALYTICAL SOLUTION - ONLY METHANE #################################
##############################################################################

'''Exponential Decay - Solving the Dynamic Equation'''
#phi(t) decreases exponentially in state 0 and increases exponentially in state 1
#i.e. phi1(t) = 1 - phi and phi2(t) = -phi

### Analytical Solution 
noise_4 = solve_analytical(noise, 0.5, dt, b)

#test range of b values
noise_1 = solve_analytical(noise, 0.5, dt, 1)
noise_025 = solve_analytical(noise, 0.5, dt, 0.25)

#%%
'''Plot Panel'''
#Plots
plot_noise(noise_4, time, name = 'b4')
#plot_noise(noise_1, time, name = 'b1')
#plot_noise(noise_025, time, name = 'b025')

#%%
'''CDF of phi'''

### Empirical cdf
fig, ax = plt.subplots()
n, bins, patches = ax.hist(noise_1.phi_a, 50, density = True, histtype = 'step', cumulative = True)
n, bins, patches = ax.hist(noise_025.phi_a, 50, density = True, histtype = 'step', cumulative = True)
n, bins, patches = ax.hist(noise_4.phi_a, 50, density = True, histtype = 'step', cumulative = True)
ax.set_xlabel(r'$[M]$')
ax.set_ylabel('Cumulative density')
plt.xlim(0, 1)
plt.legend([r'$\beta$ = 1', r'$\beta$ = 0.25', r'$\beta$ = 4'], loc = "upper left")

plt.savefig("figures/cdf_plot.pdf", bbox_inches = 'tight')

#%%
'''Theoretical PDF'''
#Generate sample pdf
samp_pdf = generate_pdf(k1, k2, b)
samp_pdf_1 = generate_pdf(k1, k2, 1)
samp_pdf_025 = generate_pdf(k1, k2, 0.25)

#%%
'''Plot PDF'''
plt.plot(samp_pdf_1.phi, samp_pdf_1.pdf, label = r'$\beta$ = 1')
plt.plot(samp_pdf_025.phi, samp_pdf_025.pdf, label = r'$\beta$ = 0.25')
plt.plot(samp_pdf.phi, samp_pdf.pdf, label = r'$\beta$ = 4')
plt.xlim(0, 1)
plt.xlabel(r'$[M]$')
plt.ylabel(r'$P[[M]]$')
plt.legend(loc = 'upper center')

plt.savefig("figures/betapdf.pdf")
#%%
##############################################################################
######### NUMERICAL SOLUTION - METHANE AND ACETATE ###########################
##############################################################################
#Run algorithm
storage = rk4_solve_ma(0.5, 0.5, dt, noise, kp = k_p, kox = k_ox, kom = k_om, e = e)

storage["dEdt"] = np.concatenate(([0], np.diff(storage.Emission)))

#%%
'''Plot Solution'''
fig, ax = plt.subplots(2, 1, figsize=(6, 5))

#ax[0].step(noise.x, noise.noise)
#ax[0].set(xlim = (0, time))
#ax[0].set(xlabel = ' ', ylabel = r'$\xi(t)$') 

ax[0].plot(storage.Time, storage.Acetate, label = 'Acetate')
ax[0].plot(storage.Time, storage.Methane, label = 'Methane')
ax[0].set(xlim = (0, 30), xlabel = ' ', ylabel = r'$[\phi]$')
ax[0].legend(loc = 'lower left')

ax[1].plot(storage.Time, storage.dEdt, label = 'Emission Rate', color = 'green')
ax[1].set(xlim = (0, 30), xlabel = 'Time [t]', ylabel = r'Emission Rate, $\frac{dE}{dt}$')

plt.savefig("figures/numSolution.pdf")

#%%
'''Empirical PDF and CDF'''


### Empirical cdf
fig, ax = plt.subplots()
n, bins, patches = ax.hist(storage.Methane, 50, density = True, histtype = 'step', cumulative = True)
n, bins, patches = ax.hist(storage.Acetate, 50, density = True, histtype = 'step', cumulative = True)
ax.set_xlabel(r'$[\phi]$')
ax.set_ylabel('Cumulative density')
plt.title("Empirical CDF of coupled Methane and Acetate")
plt.xlim(0, 1)
plt.legend(['Methane', 'Acetate'], loc = "upper left")

plt.savefig("figures/numCDF.pdf")

### Empirical pdf
fig, ax = plt.subplots()
ax.hist(storage.Methane, 50, density = True)
ax.hist(storage.Acetate, 50, density = True)
ax.set_xlabel(r'$[\phi]$')
ax.set_ylabel('Probability density')
plt.title("Empirical PDF of coupled Methane and Acetate")
plt.xlim(0, 1)
plt.legend(['Methane', 'Acetate'], loc = "upper left")

plt.savefig("figures/numPDF.pdf")


#%%
'''Simgle vs. Coupled Methane'''
## Add the cdf of measured values to this plot so we can see which system of equations gives the better estimate

### Empirical cdf
fig, ax = plt.subplots()
n, bins, patches = ax.hist(storage.Methane, 50, density = True, histtype = 'step', cumulative = True)
n, bins, patches = ax.hist(noise_4.phi_a, 50, density = True, histtype = 'step', cumulative = True)
ax.set_xlabel(r'$[M]$')
ax.set_ylabel('Cumulative density')
plt.title("Empirical CDF of methane from coupled and uncoupled solutions")
plt.xlim(0, 1)
plt.legend(['Coupled', 'Uncoupled'], loc = "upper left")

plt.savefig("figures/coupledUncoupled.pdf")

#%%
##############################################################################
######### ANALYSIS OF NUMERICAL SOLUTION FOR EMISSION ########################
##############################################################################
steps = 50

#Range of k values to test
X, Y = np.linspace(0.01, 0.99, steps), np.linspace(0.01, 0.99, steps)
Z = np.empty((steps, steps)) #Solve then Average
Z2 = np.empty((steps, steps)) #Average then Solve

Z_mbar = np.empty((steps, steps))
Z_nbar = np.empty((steps, steps))

for i in range(0, steps):
    for j in range(0, steps):
        #print("K1 is " + str(X[i]) + " and K2 is " + str(Y[j]))
        #Generate noise
        n = generate_noise(X[i], Y[j], 30, dt)
        
        #Run numerical solutions
        s = rk4_solve_ma(0.5, 0.5, dt, n, kp = k_p, kox = k_ox, kom = k_om, e = e)

        #################
        
        #Take the average dEdt
        s["dEdt"] = np.concatenate(([0], np.diff(s.Emission)))
        e_bar = s.dEdt.mean() #Mean emission per day
        e_max = s.Emission.max() #Max emission per day

        #Assign to the grid space
        Z[i, j] = e_bar
        
        #################
        
        #Take the average of methane and noise
        m_bar = s.Methane.mean()
        n_bar = s.Noise.mean()
        
        de_bar = (0.5*(k_ox*e*m_bar) - 0.5*(k_ox*e*m_bar)*n_bar)*0.1 #time step

        #Assign to the grid space
        Z2[i, j] = de_bar
        
        #################
        Z_mbar[i, j] = m_bar
        Z_nbar[i, j] = n_bar

#%%
'''Plot the dEdt bar Contour'''
#Solved and THEN averaged

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, cmap = 'OrRd', vmin = 0, vmax = 0.007)
ax.set_xlabel(r'$k_1$')
ax.set_ylabel(r'$k_2$')
plt.title("Average emissions rate")
plt.xlim(0.01, 0.99)
plt.ylim(0.01, 0.99)
plt.colorbar(contour).set_label(r'Average $\frac{dE}{dt}$')

plt.savefig("figures/heatmap.pdf")
        

#%%
'''Plot the dEdt bar Contour'''
#Averaged and THEN solved

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z2, cmap = 'OrRd', vmin = 0, vmax = 0.007)
ax.set_xlabel(r'$k_1$')
ax.set_ylabel(r'$k_2$')
plt.title("Average emissions rate")
plt.xlim(0.01, 0.99)
plt.ylim(0.01, 0.99)
plt.colorbar(contour).set_label(r'Average $\frac{dE}{dt}$')

plt.savefig("figures/heatmap2.pdf")

#%%
















