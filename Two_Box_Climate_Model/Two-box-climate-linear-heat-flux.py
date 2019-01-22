#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""This program creates a climate based off a two box Energy Balance model, where one box represents the poles, and the other the tropics.
Parameters set in this model are aimed at mimicing the global climate of the Earth. This specific model parameterizes
the meridional heat flux in a linear way, where the forcing of energy motion is dictated by the difference in 
temperature between the two boxes."""

#BEGIN CODE

import numpy
import sympy
import mpmath
from scipy.optimize import fsolve
from scipy.integrate import odeint
import math
import scipy.optimize as sp
from mpmath import mpf
mpmath.mp.dps = 15

#many values are duplicated for each box: values of '1' correspond to polar regions, '2' refers to tropical regions

c1 = 1*10**8 #constant of thermal inertia for polar region (kg/(deg K * s^2))
c2 = 1.5*10**8 #constant of thermal inertia for tropical region (kg/(deg K * s^2)
mu1 = numpy.linspace(.9,1.1,100) #Unitless factor multiplied by solar constant to vary solar intensity in Poles
mu2 = numpy.linspace(.9,1.1,100) #Unitless factor multiplied by solar constant to vary solar intensity in Tropics

I0 = 1367.0 #Solar Constant (W/m^2)
a = 2.8 #unitless parameter used to help set up a linear albedo feedback and a low temperature limit
b=.009 #parameter used to create a linear albedo feedback/ low temp limit (deg K^(-1))
sig = 5.67*10**-8 #Stephan Boltzmann Constant (Watts/(m^2 * deg K^4))
e = .66 #Unitless emissivity parameter dictating how much heat is kept on and near the earth
A = 3.3129 #Heat flux parameter derived for a linear heat focing (Watts/(m^2 * deg K))



stablebranch = []
unstablebranch = []
deepfreeze = []

def LinearAlbedoRelationship(k, mu1, mu2):
    """In order to establish a linear relationship between temperature and albedo, set b = 0 and solve the Radiation
    Balance eqution for stable solutions."""
    T1 = k[0]
    T2 = k[1]
    temp = numpy.zeros(2)
    temp[0] = (.156*mu1*I0*(1-.75) - e*sig*(T1**4))*(1.0*10**-8)
    temp[1] = (.288*mu2*I0*(1-.75) - e*sig*(T2**4))*(1.5*10**-8)
    
    return temp

for x in mu1:
    for y in mu2:
        solution = fsolve(LinearAlbedoRelationship, [350,350], args =(x,y))
        deepfreeze.append([solution[0], solution[1], x, y])
deepfreeze = numpy.array(deepfreeze)


"""To find viable solutions for upper branch in the pitchfork bifurcation structure created in this climate
model, we find only real solutions to the coupled autonomous equations, and allow the muller method to have 
guesses for a global climate of 300 and 320 degrees kelvin for the tropc and polar region"""
for x in mu1:
    for y in mu2:
        f = [lambda T1,T2: (.156*x*I0*(1-a) + .156*x*I0*b*T1 - e*sig*(T1**4)+(A*(T2 -T1))), 
             lambda T1,T2: (.288*y*I0*(1-a) + .288*y*I0*b*T2 - e*sig*(T2**4)+(A*(T1 -T2)))]
        solution = numpy.array(mpmath.findroot(f, ([300+1j,320+1j]), solver='muller', verify = False))
        if abs(mpmath.im(solution[0])) > 1e-10:
            solution[0] = numpy.nan
        if abs(mpmath.im(solution[1])) > 1e-10:
            solution[1] = numpy.nan
        stablebranch.append([mpmath.re(solution[0]), mpmath.re(solution[1]), x, y])

"""This double forloop does the same as above. Now we are looking for the structure created by the 
lower branch in the bifurcation structure. Because of this, guess solutions are 100 and 120 for the 
polar and tropic region respectively"""       
for x in mu1:
    for y in mu2:
        f = [lambda T1,T2: (.156*x*I0*(1-a) + .156*x*I0*b*T1 - e*sig*(T1**4)+(A*(T2 -T1))), 
             lambda T1,T2: (.288*y*I0*(1-a) + .288*y*I0*b*T2 - e*sig*(T2**4)+(A*(T1 -T2)))]
        solution = numpy.array(mpmath.findroot(f, ([100+1j,120+1j]), solver='muller', verify = False))
        if abs(mpmath.im(solution[0])) > 1e-10:
            solution[0] = numpy.nan
        if abs(mpmath.im(solution[1])) > 1e-10:
            solution[1] = numpy.nan
        unstablebranch.append([mpmath.re(solution[0]), mpmath.re(solution[1]), x, y])


# In[6]:


"""This section of code is dedicated solely to creating 3d images of the plots necessary in jupyters notebook. 
The plot generated is a surface in the mu1, mu2, Polar Temperature space."""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'notebook')


stablebranch = numpy.array(stablebranch)
stablebranch = stablebranch.astype(float)


unstablebranch = numpy.array(unstablebranch)
unstablebranch = unstablebranch.astype(float)

fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

xs = stablebranch[:,2] 
ys = stablebranch[:,3]
zs = stablebranch[:,0]
ax.scatter(xs, ys, zs, alpha=0.6, edgecolors='w')
xs = unstablebranch[:,2]
ys = unstablebranch[:,3]
zs = unstablebranch[:,0]
ax.scatter(xs, ys, zs, alpha=0.6, edgecolors='w')
xs = deepfreeze[:,2]
ys = deepfreeze[:,3]
zs = deepfreeze[:,0]
ax.scatter(xs, ys, zs, alpha=0.6, edgecolors='w')
ax.set_xlabel('Polar mu values')
ax.set_ylabel('Tropical mu Values')
ax.set_zlabel('Polar Equilibrium Temperatures')
plt.title('Two box climate model Linear Heat Flux Polar Equilibrium Temperatures')


# In[7]:


"""This section of code is dedicated solely to creating 3d images of the plots necessary in jupyters notebook.
The plot generated is a surface in the mu1, mu2, Tropical Temperature space."""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'notebook')


stablebranch = numpy.array(stablebranch)
stablebranch = stablebranch.astype(float)


unstablebranch = numpy.array(unstablebranch)
unstablebranch = unstablebranch.astype(float)

fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

xs = stablebranch[:,2]
ys = stablebranch[:,3]
zs = stablebranch[:,1]
ax.scatter(xs, ys, zs, alpha=0.6, edgecolors='w')
xs = unstablebranch[:,2]
ys = unstablebranch[:,3]
zs = unstablebranch[:,1]
ax.scatter(xs, ys, zs, alpha=0.6, edgecolors='w')
xs = deepfreeze[:,2]
ys = deepfreeze[:,3]
zs = deepfreeze[:,1]
ax.scatter(xs, ys, zs, alpha=0.6, edgecolors='w')
ax.set_xlabel('Polar mu values')
ax.set_ylabel('Tropical mu Values')
ax.set_zlabel('Tropical Equilibrium Temperatures')
plt.title('Two box climate model Linear Heat Flux Tropical Equilibrium Temperatures')


# In[ ]:




