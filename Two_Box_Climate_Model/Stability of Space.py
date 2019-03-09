#!/usr/bin/env python
# coding: utf-8

# In[22]:


"""In creation of the reciprical heat flux we find that the upper branch is an unstable solution. Because of this,
we would like to investigate how changes in the parameters in the two-box reciporical heat flux model can be altered
to produce stable configurations of the global climate. At the point of creation of this code, a linearized Jacobian
has been created from the coupled autonomous differential equations. This linearized version has the same stability 
as the original. Because of this, finding where the linearized version is stable will tell us where the original is
stable as well.


NOTE: C1 and C2 are kept out of these equations because multiplying by such a small number makes the computational
method of finding roots eventually start to register significant imaginary parts of the solution (i*10^-6). This has
been cross referenced with other methods for root finding and solutions DO NOT have imaginary parts. Also, 
we are only testing the upper branch solutions which appear to be the best hope for finding a stable configuration.

Aodhan Sweeney
February 21st 2019
Stability Space"""

"""The first object to vary is the scaling constant for both the polar and tropical box."""

"""The Next object we will change is the Heat flux constant"""

import numpy as np
import mpmath as mp
import math as m

def jacobian_stability_analysis(J11, J12, J21, J22):
    f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
    z1 = (mp.findroot(f, ([100+1j]), solver='muller', verify = False))
    z2 = (mp.findroot(f, ([-100+1j]), solver='muller', verify = False))
    return np.array([z1,z2])

if __name__ == '__main__':
    b = .009  #K^-1 for the linear albedo relationship (really called b i just dont want to confuse it with above variable)
    e = .64 #emissivity parameter 
    a = 2.8
    A = 600 #heat flux parameter
    polar_scalar = .156 # scale factor for polar region 
    tropic_scalar = .288 # scalar factor for tropic region
    mu1 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (polar)
    mu2 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (tropic)
    I0 = 1367.0 # Solar constant at top of atmosphere in W/m^2
    sig = 5.67*10**-8 # Stephan Boltzmann Constant
        
    lambdas = []
    temperatures = []
    
    solar_constants = np.linspace(.0001, 1.57, 1000)
    for x in solar_constants:
        p = (np.pi - 2*x - m.sin(2*x))/(4*np.pi*(1-m.sin(x)))
        t = (x + .5*m.sin(2*x))/(2*np.pi*m.sin(x))
        f = [lambda T1,T2: (p*1.0*I0*(1-a) + p*1.0*I0*b*T1 - e*sig*(T1**4)+(A/(T2 -T1))), 
             lambda T1,T2: (t*1.0*I0*(1-a) + t*1.0*I0*b*T2 - e*sig*(T2**4)+(A/(T1 -T2)))]
        solution = np.array(mp.findroot(f, ([400+1j,405+1j]), solver='muller', verify = False))
        if abs(mp.im(solution[0])) > 1e-10:
            solution[0] = np.nan
        if abs(mp.im(solution[1])) > 1e-10:
            solution[1] = np.nan
        #This sets up the T1 and T2 needed for the stability analysis
        x = mp.re(solution[0])
        y = mp.re(solution[1])  

        eq_temp = np.array([x, y, p, t])
        eq_temp = eq_temp.astype(float)
        temperatures.append(eq_temp)

    temperatures = np.array(temperatures)
    temperatures = temperatures.astype(float)
    
    select = ~np.isnan(temperatures[:,0])
    temperatures = temperatures[select]
    
    select = ~np.isnan(temperatures[:,1])
    temperatures = temperatures[select]
    
        
    for eq_temp in temperatures:
        J11 = (eq_temp[2]*mu1*I0*b - 4*e*sig*(eq_temp[0]**3)+(A/(eq_temp[1] - eq_temp[0])**2))
        J22 = (eq_temp[3]*mu2*I0*b - 4*e*sig*(eq_temp[1]**3)+(A/(eq_temp[0] - eq_temp[1])**2))
        J12 = -A/((eq_temp[1] - eq_temp[0])**2)
        J21 = -A/((eq_temp[0] - eq_temp[1])**2)
        stability = jacobian_stability_analysis(J11, J12, J21, J22)
        lambdas.append([stability, eq_temp[0], eq_temp[1], eq_temp[2], eq_temp[3]])

    
    
    lambdas = np.array(lambdas)
    for x in lambdas:
        print(' polar temp: ',x[1], '\n tropic temp: ', x[2], '\n polar constant, tropic constant : ', x[3], x[4], '\n', x[0], '\n')


# In[38]:


"""The Next object we will change is the Heat flux constant"""

import numpy as np
import mpmath as mp


def jacobian_stability_analysis(J11, J12, J21, J22):
    f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
    z1 = (mp.findroot(f, ([100+1j]), solver='muller', verify = False))
    z2 = (mp.findroot(f, ([-100+1j]), solver='muller', verify = False))
    return np.array([z1,z2])

if __name__ == '__main__':
    b = .009  #K^-1 for the linear albedo relationship 
    e = .64 #emissivity parameter 
    a = 2.8
    A = 600 #heat flux parameter
    polar_scalar = .1551 # scale factor for polar region 
    tropic_scalar = .2893 # scalar factor for tropic region
    mu1 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (polar)
    mu2 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (tropic)
    I0 = 1367.0 # Solar constant at top of atmosphere in W/m^2
    sig = 5.67*10**-8 # Stephan Boltzmann Constant
        
    lambdas = []
    temperatures = []
    
    heat_flux_constants = np.linspace(0, 10000, 500)
    for A in heat_flux_constants:
        f = [lambda T1,T2: (.156*1.0*I0*(1-a) + .156*1.0*I0*b*T1 - e*sig*(T1**4)+(A/(T2 -T1))), 
             lambda T1,T2: (.288*1.0*I0*(1-a) + .288*1.0*I0*b*T2 - e*sig*(T2**4)+(A/(T1 -T2)))]
        solution = np.array(mp.findroot(f, ([400+1j,405+1j]), solver='muller', verify = False))
        if abs(mp.im(solution[0])) > 1e-10:
            solution[0] = np.nan
        if abs(mp.im(solution[1])) > 1e-10:
            solution[1] = np.nan
        #This sets up the T1 and T2 needed for the stability analysis
        x = mp.re(solution[0])
        y = mp.re(solution[1])  
        
        eq_temp = np.array([x, y, A])
        eq_temp = eq_temp.astype(float)
        temperatures.append(eq_temp)

    temperatures = np.array(temperatures)
    temperatures = temperatures.astype(float)
    
    select = ~np.isnan(temperatures[:,0])
    temperatures = temperatures[select]
    
    select = ~np.isnan(temperatures[:,1])
    temperatures = temperatures[select]
    
        
    for eq_temp in temperatures:
        J11 = (polar_scalar*mu1*I0*b - 4*e*sig*(eq_temp[0]**3)+(eq_temp[2]/(eq_temp[1] - eq_temp[0])**2))
        J22 = (tropic_scalar*mu2*I0*b - 4*e*sig*(eq_temp[1]**3)+(eq_temp[2]/(eq_temp[0] - eq_temp[1])**2))
        J12 = -eq_temp[2]/((eq_temp[1] - eq_temp[0])**2)
        J21 = -eq_temp[2]/((eq_temp[0] - eq_temp[1])**2)
        stability = jacobian_stability_analysis(J11, J12, J21, J22)
        lambdas.append([stability, eq_temp[0], eq_temp[1], eq_temp[2]])

    
    
    lambdas = np.array(lambdas)
    for x in lambdas:
        print(' polar temp: ',x[1], '\n tropic temp: ', x[2], '\n Heat Flux Parameter: ', x[3], '\n', x[0], '\n')

"""A change in the heat flux parameter may induce stable structures for the bifurcation fold. If the heat flux
parameter is sufficiently small we may find solutions that fit a stable configuration. With the rest of the parameters
in each state equation constant, the change in heat flux in too dramatic to model any real life scenario (non lin
heat flux of 16.5). Because of the appeal of finding a realistic stable configuration, a change in the emissivity is
also tested in tandum with a reduction to the heat flux. """


# In[4]:


"""The next object we will change is the emissivity parameter."""


import numpy as np
import mpmath as mp


def jacobian_stability_analysis(J11, J12, J21, J22):
    f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
    z1 = (mp.findroot(f, ([100+1j]), solver='muller', verify = False))
    z2 = (mp.findroot(f, ([-100+1j]), solver='muller', verify = False))
    return np.array([z1,z2])

if __name__ == '__main__':
    b = .009  #K^-1 for the linear albedo relationship (really called b i just dont want to confuse it with above variable)
    e = .64 #emissivity parameter 
    a = 2.8
    A = 600 #heat flux parameter
    polar_scalar = .156 # scale factor for polar region 
    tropic_scalar = .288 # scalar factor for tropic region
    mu1 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (polar)
    mu2 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (tropic)
    I0 = 1367.0 # Solar constant at top of atmosphere in W/m^2
    sig = 5.67*10**-8 # Stephan Boltzmann Constant
        
    lambdas = []
    temperatures = []
    
    emissivities = np.linspace(.3, .8, 400)
    for e in emissivities:
        f = [lambda T1,T2: (.156*1.0*I0*(1-a) + .156*1.0*I0*b*T1 - e*sig*(T1**4)+(A/(T2 -T1))), 
             lambda T1,T2: (.288*1.0*I0*(1-a) + .288*1.0*I0*b*T2 - e*sig*(T2**4)+(A/(T1 -T2)))]
        solution = np.array(mp.findroot(f, ([400+1j,405+1j]), solver='muller', verify = False))
        if abs(mp.im(solution[0])) > 1e-10:
            solution[0] = np.nan
        if abs(mp.im(solution[1])) > 1e-10:
            solution[1] = np.nan
        #This sets up the T1 and T2 needed for the stability analysis
        x = mp.re(solution[0])
        y = mp.re(solution[1])  
        
        eq_temp = np.array([x, y, e])
        eq_temp = eq_temp.astype(float)
        temperatures.append(eq_temp)

    temperatures = np.array(temperatures)
    temperatures = temperatures.astype(float)
    
    select = ~np.isnan(temperatures[:,0])
    temperatures = temperatures[select]
    
    select = ~np.isnan(temperatures[:,1])
    temperatures = temperatures[select]
    
    for eq_temp in temperatures:
        J11 = (polar_scalar*mu1*I0*b - 4*eq_temp[2]*sig*(eq_temp[0]**3)+(A/(eq_temp[1] - eq_temp[0])**2))
        J22 = (tropic_scalar*mu2*I0*b - 4*eq_temp[2]*sig*(eq_temp[1]**3)+(A/(eq_temp[0] - eq_temp[1])**2))
        J12 = -A/((eq_temp[1] - eq_temp[0])**2)
        J21 = -A/((eq_temp[0] - eq_temp[1])**2)
        stability = jacobian_stability_analysis(J11, J12, J21, J22)
        lambdas.append([stability, eq_temp[0], eq_temp[1], eq_temp[2]])

    
    
    lambdas = np.array(lambdas)
    for x in lambdas:
        print(' polar temp: ',x[1], '\n tropic temp: ', x[2], '\n emissivity: ', x[3],'\n', x[0], '\n')


# In[25]:


"""The next object we will change is the ~a~ feedback parameter."""


import numpy as np
import mpmath as mp


def jacobian_stability_analysis(J11, J12, J21, J22):
    f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
    z1 = (mp.findroot(f, ([100+1j]), solver='muller', verify = False))
    z2 = (mp.findroot(f, ([-100+1j]), solver='muller', verify = False))
    return np.array([z1,z2])

if __name__ == '__main__':
    b = .009  #K^-1 for the linear albedo relationship (really called b i just dont want to confuse it with above variable)
    e = .64 #emissivity parameter 
    #a = 2.8
    A = 600 #heat flux parameter
    polar_scalar = .156 # scale factor for polar region 
    tropic_scalar = .288 # scalar factor for tropic region
    mu1 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (polar)
    mu2 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (tropic)
    I0 = 1367.0 # Solar constant at top of atmosphere in W/m^2
    sig = 5.67*10**-8 # Stephan Boltzmann Constant
        
    lambdas = []
    temperatures = []
    
    feedbacks = np.linspace(.3, 5, 400)
    for a in feedbacks:
        f = [lambda T1,T2: (.156*1.0*I0*(1-a) + .156*1.0*I0*b*T1 - e*sig*(T1**4)+(A/(T2 -T1))), 
             lambda T1,T2: (.288*1.0*I0*(1-a) + .288*1.0*I0*b*T2 - e*sig*(T2**4)+(A/(T1 -T2)))]
        solution = np.array(mp.findroot(f, ([400+1j,405+1j]), solver='muller', verify = False))
        if abs(mp.im(solution[0])) > 1e-10:
            solution[0] = np.nan
        if abs(mp.im(solution[1])) > 1e-10:
            solution[1] = np.nan
        #This sets up the T1 and T2 needed for the stability analysis
        x = mp.re(solution[0])
        y = mp.re(solution[1])  
        
        eq_temp = np.array([x, y, a])
        eq_temp = eq_temp.astype(float)
        temperatures.append(eq_temp)

    temperatures = np.array(temperatures)
    temperatures = temperatures.astype(float)
    
    select = ~np.isnan(temperatures[:,0])
    temperatures = temperatures[select]
    
    select = ~np.isnan(temperatures[:,1])
    temperatures = temperatures[select]
    
    for eq_temp in temperatures:
        J11 = (polar_scalar*mu1*I0*b - 4*e*sig*(eq_temp[0]**3)+(A/(eq_temp[1] - eq_temp[0])**2))
        J22 = (tropic_scalar*mu2*I0*b - 4*e*sig*(eq_temp[1]**3)+(A/(eq_temp[0] - eq_temp[1])**2))
        J12 = -A/((eq_temp[1] - eq_temp[0])**2)
        J21 = -A/((eq_temp[0] - eq_temp[1])**2)
        stability = jacobian_stability_analysis(J11, J12, J21, J22)
        lambdas.append([stability, eq_temp[0], eq_temp[1], eq_temp[2]])

    
    
    lambdas = np.array(lambdas)
    for x in lambdas:
        print(' polar temp: ',x[1], '\n tropic temp: ', x[2], '\n feebacks constant -a- : ', x[3], '\n', x[0], '\n')


# In[40]:


"""The next object we will change is the ~b~ feedback parameter."""


import numpy as np
import mpmath as mp

def jacobian_stability_analysis(J11, J12, J21, J22):
    f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
    z1 = (mp.findroot(f, ([100+1j]), solver='muller', verify = False))
    z2 = (mp.findroot(f, ([-100+1j]), solver='muller', verify = False))
    return np.array([z1,z2])

if __name__ == '__main__':
    b = .009  #K^-1 for the linear albedo relationship (really called b i just dont want to confuse it with above variable)
    e = .64 #emissivity parameter 
    a = 2.8
    A = 600 #heat flux parameter
    polar_scalar = .156 # scale factor for polar region 
    tropic_scalar = .288 # scalar factor for tropic region
    mu1 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (polar)
    mu2 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (tropic)
    I0 = 1367.0 # Solar constant at top of atmosphere in W/m^2
    sig = 5.67*10**-8 # Stephan Boltzmann Constant
        
    lambdas = []
    temperatures = []
    
    feedbacks = np.linspace(.0001, .1, 1000)
    for b in feedbacks:
        f = [lambda T1,T2: (.156*1.0*I0*(1-a) + .156*1.0*I0*b*T1 - e*sig*(T1**4)+(A/(T2 -T1))), 
             lambda T1,T2: (.288*1.0*I0*(1-a) + .288*1.0*I0*b*T2 - e*sig*(T2**4)+(A/(T1 -T2)))]
        solution = np.array(mp.findroot(f, ([400+1j,405+1j]), solver='muller', verify = False))
        if abs(mp.im(solution[0])) > 1e-10:
            solution[0] = np.nan
        if abs(mp.im(solution[1])) > 1e-10:
            solution[1] = np.nan
        #This sets up the T1 and T2 needed for the stability analysis
        x = mp.re(solution[0])
        y = mp.re(solution[1])  
        
        eq_temp = np.array([x, y, b])
        eq_temp = eq_temp.astype(float)
        temperatures.append(eq_temp)

    temperatures = np.array(temperatures)
    temperatures = temperatures.astype(float)
    
    select = ~np.isnan(temperatures[:,0])
    temperatures = temperatures[select]
    
    select = ~np.isnan(temperatures[:,1])
    temperatures = temperatures[select]
    
    for eq_temp in temperatures:
        J11 = (polar_scalar*mu1*I0*eq_temp[2] - 4*e*sig*(eq_temp[0]**3) + (A/(eq_temp[1] - eq_temp[0])**2))
        J22 = (tropic_scalar*mu2*I0*eq_temp[2] - 4*e*sig*(eq_temp[1]**3) + (A/(eq_temp[0] - eq_temp[1])**2))
        J12 = -A/((eq_temp[1] - eq_temp[0])**2)
        J21 = -A/((eq_temp[0] - eq_temp[1])**2)
        stability = jacobian_stability_analysis(J11, J12, J21, J22)
        lambdas.append([stability, eq_temp[0], eq_temp[1], eq_temp[2]])

    
    
    lambdas = np.array(lambdas)

    for x in lambdas:
        print(' polar temp: ',x[1], '\n tropic temp: ', x[2], '\n feebacks constant -b-: ', x[3], '\n', x[0], '\n')


# In[368]:


"""Plotting streamfunctions for Non-Linear Heat Flux for present day values of solar intensity"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#List of Constants
b = .009  #K^-1 for the linear albedo relationship 
e = .64 #emissivity parameter 
A = 600 #heat flux parameter
polar_scalar = .156 # scale factor for polar region 
tropic_scalar = .288 # scalar factor for tropic region
a = 2.8
#T1 = 305.23546209 # Equilibrium temperature for the polar region
#T2 = 311.69945035 # Equilirbium temperature for the tropic region
mu1 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (polar)
mu2 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (tropic)
I0 = 1367.0 # Solar constant at top of atmosphere in W/m^2
sig = 5.67*10**-8 # Stephan Boltzmann Constant   
c1 = 1e8 # Specific heat for 
c2 = 1.5e8   
    

    
if __name__ == '__main__':
    
    T2, T1  = np.mgrid[250:290:1000j, 240:280:1000j] #Grid created so that stream function can be called
    
    U = (.156*mu1*I0*(1-a) + .156*mu1*I0*b*T1 - e*sig*(T1**4) + (A/(T2 -T1)))/c1
    V = (.288*mu2*I0*(1-a) + .288*mu2*I0*b*T2 - e*sig*(T2**4) + (A/(T1 -T2)))/c2

    fig = plt.figure(figsize = (20,30))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1,2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(T1, T2, U, V, density=[2,2])
    plt.plot([273.312], [283.046], marker='o', markersize='10', color='black', label='Unstable Sadle-Point')
    plt.plot([243.487], [257.123], marker='o', markersize='10', color='red', label='Unstable Node (Source)')
    x_array = range(200,300)
    y_array = range(200,300)
    #plt.plot(x_array, y_array)
    ax0.set_title('Stability Stream Plot: Non-Linear Heat FLux')
    ax0.set_xlabel('Polar Temperatures (degree K)')
    ax0.set_ylabel('Tropical Temperatures (degree K)')
    ax0.legend(loc=2)
    plt.savefig('/Users/aodhan/Desktop/Stability_Non_Lin.png', bbox_inches='tight', dpi=600)
    plt.show()
"""Two equilibriums, one saddle point stability and the other is an unstable node. A key point of interest is the 
T1 = T2 line. It seems that right below the line connecting the equilibrium points, we see normal node behavior. 
Crossing below the T1=T2 line we are then in the regime of initial conditions where the polar box has a higher 
temeperature than the tropcial box. If this happens, then the polar box starts to transfer heat to the tropical box 
and it appears that the solution is different."""


# In[9]:


"""Plotting streamfunctions for Linear Heat Flux."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#X and Y correspond to Polar and Tropical regions respectfully\
b = .009  #K^-1 for the linear albedo relationship (really called b i just dont want to confuse it with above variable)
e = .64 #emissivity parameter 
A = 3 #heat flux parameter
polar_scalar = .156 # scale factor for polar region 
tropic_scalar = .288 # scalar factor for tropic region
a = 2.8

mu1 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (polar)
mu2 = 1.0 # a sample mu configuration of upper branch corresponding to the temperature equilibriums (tropic)
I0 = 1367.0 # Solar constant at top of atmosphere in W/m^2
sig = 5.67*10**-8 # Stephan Boltzmann Constant   
c1 = 1e8 # Specific heat for 
c2 = 1.5e8   
    
    
T2, T1  = np.mgrid[250:305:500j, 240:280:500j] #Grid created so that stream function can be called
U = (.156*mu1*I0*(1-a) + .156*mu1*I0*b*T1 - e*sig*(T1**4) + (A*(T2 -T1)))/c1
V = (.288*mu2*I0*(1-a) + .288*mu2*I0*b*T2 - e*sig*(T2**4) + (A*(T1 -T2)))/c2

fig = plt.figure(figsize = (20,30))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1,2])

#  Varying density along a streamline
ax1 = fig.add_subplot(gs[0, 0])
ax1.streamplot(T1, T2, U, V, density=[2,2])
plt.plot([272.82], [293.39], marker='o', markersize='10', color='black', label='Stable Node (Sink)')
plt.plot([242.8], [257.6], marker='o', markersize='10', color='red', label= 'Unstable Saddle-Point')

ax1.set_title('Stability Stream Plot: Linear Heat Flux')
ax1.set_xlabel('Polar Temperatures (degree K)')
ax1.set_ylabel('Tropical Temperatures (degree K)')
ax1.legend(loc=2)
plt.savefig('/Users/aodhan/Desktop/Stability_Linear.png', bbox_inches='tight', dpi=600)

plt.show()


# In[ ]:




