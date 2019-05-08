#!/usr/bin/env python
# coding: utf-8

# In[7]:


"""This code recreates the 1978 zero dimensional radiation based climate model created by Fraedrich. In hopes of
creating a two box climate scheme based on this model, it was crucial to recrete the foundational model of zero
dimensional radiation based climate models."""

#BEGIN CODE

import sympy
import mpmath
import numpy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy.ma as ma
from scipy.optimize import root



mu = numpy.linspace(.94,1.04,500) #Unitless value that allows us to scale solar constant
I0 = 1367.0 #solar constant (W/m^2)
a = 2.8 #Unitless value helping to create linear albedo feedback
b=.009 #value created to help make linear albedo feedback and is multiplied by temperature (deg K^-1)
sig = 5.67*10**-8 #Botlzmann constant  (W/(m^2*k^4))
e = .69 #Unitless emissivity parameter
T = sympy.Symbol('T') #Temperature value



deepfreeze = numpy.array([])
stablebranch = numpy.array([])
unstablebranch = numpy.array([])





def LinearAlbedoRelationship(T, mu):
    """We want to establish a linear relationship between temperature and albedo. We can set b = 0 and solve the Radiation
    Balance eqution for stable solutions to the problem. This is the lower limit of the solutions. This bound corresponds to the glaciel period deep freeze"""

    y = (.25*mu*I0*(1-.75) - e*sig*(T**4))*(1.0*10**-8)
    return y


def RadiationBalance(T, mu):
    """Differential Equation describing global radiation budget, Assuming constant impact radiation everywhere."""
    y = (.25*mu*I0*(1-a) + .25*mu*I0*b*T - e*sig*(T**4))*(1.0*10**-8)
    return y

deepfreeze = numpy.concatenate([fsolve(LinearAlbedoRelationship, 10, x) for x in mu])


for x in mu:
    """To find solutions to the upper branch of the pitchfork bifurcation we choose guess temperature equilibrium
    values that are high compared to the global average."""
    soln = mpmath.findroot(lambda T: RadiationBalance(T,x),(290,200,310), solver='muller')
    if mpmath.im(soln) != 0:
        soln = numpy.nan
    stablebranch = numpy.append(stablebranch, soln)
stablebranch = numpy.array(stablebranch)


for x in mu:
    """Finding solutions to the lower branch of the pitchfork bifurcation, we choose guess solutions that are
    low compared to the global average. We are looking for only physically realistic roots."""
    soln = mpmath.findroot(lambda T: RadiationBalance(T,x),(200,210,220), solver='muller')
    if mpmath.im(soln) != 0:
        soln = numpy.nan
    unstablebranch = numpy.append(unstablebranch, soln)
unstablebranch = numpy.array(unstablebranch)


#This section of the code is dedicated to making the plot of the 1978 Fraedrich paper zero dimensional climate model

plt.plot(mu, stablebranch, color='red', label = 'Stable Branch')
plt.plot(mu,unstablebranch, color= 'orange', label = 'Unstable Branch')
plt.plot(mu, deepfreeze, color = 'blue', label = 'Deep Freeze Solution')
plt.xlabel('Relative Intensity of Solar Constant ')
plt.ylabel("Globally Averaged Temperatures (k)")
plt.title('Fraedrich 1978 Zero Dimmensional Climate Model')
plt.legend()
