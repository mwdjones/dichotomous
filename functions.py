
'''Imports'''
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#For solving systems of equations
import sympy as sp
from sympy.solvers import solve

#############################
'''Model Functions'''

def generate_noise(k1, k2, num, step):
    #Takes in switching rates and generates symmetric dichotomous noise for a specified time period and step. 
    #Inputs: k1, k2 - The switching rates from state 1 to 0 and 0 to 1 respectively
    #        num - length of time 
    #        step - time step 
    #Outputs: A pandas dataframe containing two columns, one with the time and one with the noise value (either -1 or 1)
    
    #Initialize the noise
    switch = []

    #Generate random switching
    for i in range(0, num+50): #buffer
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

def solve_analytical(noise, conc_init, step, beta):
    #Takes in the noise generated by generate_noise and solved the methane equations analytically 
    #for an initial concentration.
    #Input: n - pandas dataframe containing two columns, 'x'; the time, 'noise'; the binary noise value
    #       conc_init - the initial concentration of methane 
    #       step - the time step (this should match the time step from generate_noise)
    #       beta - the beta value, i.e. the nondimensionalized constant
    #Output: noise - the input noise dataframe with four additional columns for values of f_phi, g_phi, 
    #        dphi_a, and phi_a
    
    n = pd.DataFrame({'x': noise.x, 'noise': noise.noise})
    n['f_phi'] = np.zeros(len(n.noise))
    n['g_phi'] = np.zeros(len(n.noise))
    n['dphi_a'] = np.zeros(len(n.noise))
    n['phi_a'] = np.zeros(len(n.noise))
    
    #Initial Concentration
    n.phi_a[0] = conc_init
    
    for i in range(1, len(n.phi_a)):
        n.f_phi[i] = 0.5*(beta*(1 - n.phi_a[i-1]) - n.phi_a[i-1])
        n.g_phi[i] = 0.5*(beta*(1 - n.phi_a[i-1]) + n.phi_a[i-1])
        n.dphi_a[i] = (n.f_phi[i] + n.g_phi[i]*n.noise[i])*step
        n.phi_a[i] = n.phi_a[i-1] + n.dphi_a[i]
        
    return n

def plot_noise(nplot, time, name):
    fig, ax = plt.subplots(3, 1, figsize=(7, 11))
    
    ax[0].step(nplot.x, nplot.noise)
    ax[0].set(xlim = (0, time))
    ax[0].set(xlabel = ' ', ylabel = r'$\xi(t)$') 
    
    ax[1].plot(nplot.x, nplot.phi_a, zorder = 0)
    ax[1].set(xlim = (0, time))
    ax[1].set(xlabel = ' ', ylabel = r'$[M]$')
    
    ax[2].plot(nplot.x, nplot.dphi_a, zorder = 0)
    ax[2].set(ylim = (-0.3, 0.3), xlim = (0, time))
    ax[2].set(xlabel = 'Time [t]', ylabel = r'$\frac{d[M]}{dt}$')
    
    plt.show()

def generate_pdf(k1, k2, b):
    
    #Generate range of phi values
    pdf = pd.DataFrame({'phi': np.arange(0, 1, step = 0.01)})
    
    #Solve pdf equation for range of phi values
    term1 = [math.pow(value, k2) for value in pdf.phi]
    term2 = [math.pow(1 - value, k1/b) for value in pdf.phi]
    coeff = (pdf.phi + b*(1 - pdf.phi))/(pdf.phi*b*(1 - pdf.phi))
    
    pdf['pdf'] = coeff*term1*term2 
    
    #Normalize over the total sum so that the probabilities sum to one
    pdf_sum = np.sum(pdf.pdf)
    
    pdf['pdf'] = pdf.pdf/pdf_sum
    
    return pdf

def ma_equations_dim(a, m, noise, kp, kox, kom, e):
    #Takes in all values necessary for solving the coupled equations for a given acetate and metane concetration
    #and then returns a 3x1 vector containing the values of d[m]dt and d[a]dt and dEdt
    #Input: a - acetate concentration
    #       m - methane concentration 
    #       k_p - Methane production rate constant (1/t)
    #       k_ox - Methane oxidation production rate constant (1/t)
    #       k_om - Acetate fermentation rate constant (1/t)
    #       e - Emission constant (dimensionless)
    #       noise - the current value for dichotomous noise, either 1 or -1. 
    #Output: sol - a vector containing the values of d[m]dt and d[a]dt and dEdt
    
    sol = []
    
    dm = 0.5*(kp*a*(1-m) - kox*m*(1-e)) + 0.5*(kp*a*(1-m) + kox*m*(1-e))*noise
    da = 0.5*(kom*(1-a) - kp*a*(1-m)) - 0.5*(kom*(1-a) + kp*a*(1 - m))*noise
    de = 0.5*(kox*e*m) - 0.5*(kox*e*m)*noise
    
    sol.append(da)
    sol.append(dm)
    sol.append(de)
    
    return np.array(sol)


def rk4_solve_ma(m_init, a_init, step, noise, kp, kox, kom, e):
    
    s = pd.DataFrame({'Time': noise.x,
                      'Noise': noise.noise, 
                      'Acetate': np.zeros(len(noise.x)), 
                      'Methane': np.zeros(len(noise.x)),
                      'Emission': np.zeros(len(noise.x))})
    
    #Set initial conditions - Initial emission is left at 0
    s.Acetate[0], s.Methane[0],  = a_init, m_init
    
    for i in range(1, len(noise)):
        v = np.array([s.Acetate[i-1], s.Methane[i-1], s.Emission[i-1]])
        v1 = ma_equations_dim(v[0], v[1], s.Noise[i-1], kp = kp, kox = kox, kom = kom, e = e)
        tv = v + ((0.5*step)*v1)
        v2 = ma_equations_dim(tv[0], tv[1], s.Noise[i-1], kp = kp, kox = kox, kom = kom, e = e)
        tv = v + ((0.5*step)*v2)
        v3 = ma_equations_dim(tv[0], tv[1], s.Noise[i-1], kp = kp, kox = kox, kom = kom, e = e)
        tv = v + step*v3
        v4 = ma_equations_dim(tv[0], tv[1], s.Noise[i-1], kp = kp, kox = kox, kom = kom, e = e)
        
        solution = v + (step/6)*(v1 + 2*v2 + 2*v3 + v4)
        
        #Assign values back to dataframe
        s.Acetate[i], s.Methane[i], s.Emission[i] = solution[0], solution[1], solution[2]   
        
    return s


##############################
'''Analysis Functions'''

def switch_param(series, thresh):
    #A function that will take a time series and a threshold and return the switching 
    #constants as determined by the time spent above and below the threshold
    #Inputs: series - this is the series that will parametrized. In this case a WTE time series (array like)
    #        thresh - the threshod that will be used to compute the crossing (int)
    #Output: k1_p (1/t1), k2_p (1/t2) - the switching parameters 
    #        N - the series approximated as a dichotomous noise simulation 
    
    #Calculate the time above and below
    t1 = len(series[series >= thresh])
    t2 = len(series[series < thresh])
    
    #Make sure the numbers make sense
    if(t1 + t2 != len(series)):
        return np.NAN
    
    #Create the new time series
    approx = np.ones(len(series))
    approx[series < thresh] = -1
    
    #Return
    return 1/t1, 1/t2, approx

def calc_rates(noise):
    tot = len(noise)
    s1 = len(noise[noise > 0]) #state 1 is above zero (1)
    s2 = len(noise[noise < 0]) #state 2 is below zero (-1)

    p1 = s1/tot #probability of being in state 1
    p2 = s2/tot #probability of being in state 2

    #Solve system of equation
    #P1(k1 + k2) = k2 --> P1k1 + P1k2 = k2 --> P1k1 + (P1 - 1)k2 = 0
    #P2(k1 + k2) = k1 --> P2k1 + P2k2 = k1 --> (P2 - 1)k1 + P2k2 = 0

    #Solution using nummpy -- always gives trivial solution
    #a = np.array([[p1, (p1 - 1)], [(p2 - 1), p2]])
    #b = np.array([0, 0])
    #sol = np.linalg.solve(a,b)

    #Solution using sympy
    k1, k2 = sp.symbols('k1, k2')
    eq1= sp.Eq(p1*k1 + (p1 - 1)*k2) 
    eq2 = sp.Eq((p2 - 1)*k1 + p2*k2)
    sol = solve([eq1,eq2], k1, dict = True)

    #Solution will always give a relationship between x and y
    #fix x (k1) as 1
    eq21 = sp.Eq(sol[0][k1] - k1)
    eq22 = sp.Eq(k1 - 1)
    sol2 = solve([eq21, eq22], dict = True)

    #Pull out switching constants
    
    return sol2[0][k1], sol2[0][k2]

def downsample(df, col, step):
    n = len(df)/step

    #Resampled indexes
    xresamp = np.arange(min(df[col]), max(df[col])+step, step)

    #set og index
    df.set_index(col, inplace =True)

    #new df
    df2 = df.reindex(xresamp)

    #interpolate
    df_resampled = df2.interpolate('nearest').loc[xresamp].reset_index(drop = False)

    return df_resampled

def ncrosses(series, thresh):
    count = 0
    # Iterate over the steps array
    for i in range(1, len(series)):
        # Condition to check that the
        # graph crosses the origin.
        if ((series[i-1] < thresh and series[i] >= thresh) or (series[i-1] > thresh and series[i] <= thresh)):
            count += 1
 
    return count