#! /usr/bin/env python
"""In creation of the reciprical heat flux we find that the upper branch is an unstable solution. Because of this,
we would like to investigate how changes in the parameters in the two-box reciporical heat flux model can be altered
to produce stable configurations of the global climate. At the point of creation of this code, a linearized Jacobian
has been created from the coupled autonomous differential equations. This linearized version has the same stability 
as the original. Because of this, finding where the linearized version is stable will tell us where the original is
stable as well. The data output is saved as a .npy file
NOTE: C1 and C2 are kept out of these equations because multiplying by such a small number makes the computational
method of finding roots eventually start to register significant imaginary parts of the solution (i*10^-6). This has
been cross referenced with other methods for root finding and solutions DO NOT have imaginary parts. Also, 
we are only testing the upper branch solutions which appear to be the best hope for finding a stable configuration.


Aodhan Sweeney
February 21st 2019
Stability Space
"""

import numpy as np
import mpmath as mp
import math as m

def jacobian_stability_analysis(J11, J12, J21, J22):
    """jacobian_stability_analysis is a function that performs the stability analysis of each equilibria found.
    For each equilibria, a jacobian is created, the eigenvalues are then found of this jacobian and returned 
    by this function."""
    f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
    z1 = (mp.findroot(f, ([100+1j]), solver='muller', verify = False))
    z2 = (mp.findroot(f, ([-100+1j]), solver='muller', verify = False))
    eigenvalues = np.array[z1, z2]

    return eigenvalues.astype(float)

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
    
    solar_constants = np.linspace(.0001, 1.57, 50)
    heat_fluxes = np.linspace(0,10000, 100)
    emissivities = np.linspace(.3, .8, 50)
    a_feedback = np.linspace(1, 5, 50)
    b_feedback = np.linspace(.0001, .1, 50)
    for theta in solar_constants:
        for A in heat_fluxes:
            for e in emissivities:
                for a in a_feedback:
                    for b in b_feedback:
                        p = (np.pi - 2*theta - m.sin(2*theta))/(4*np.pi*(1-m.sin(theta)))
                        t = (theta + .5*m.sin(2*theta))/(2*np.pi*m.sin(theta))
                        f = [lambda T1,T2: (p*mu1*I0*(1-a) + p*mu1*I0*b*T1 - e*sig*(T1**4)+(A/(T2 -T1))), 
                             lambda T1,T2: (t*mu2*I0*(1-a) + t*mu2*I0*b*T2 - e*sig*(T2**4)+(A/(T1 -T2)))]
                        solution = np.array(mp.findroot(f, ([300+1j,405+1j]), solver='muller', verify = False))
                        if abs(mp.im(solution[0])) > 1e-10:
                            solution[0] = np.nan
                        if abs(mp.im(solution[1])) > 1e-10:
                            solution[1] = np.nan
                        #This sets up the T1 and T2 needed for the stability analysis
                        x = mp.re(solution[0])
                        y = mp.re(solution[1])  

                        eq_temp = np.array([x, y, p, t, A, e, a, b])
                        eq_temp = eq_temp.astype(float)
                        temperatures.append(eq_temp)

    temperatures = np.array(temperatures)
    temperatures = temperatures.astype(float)
    
    select = ~np.isnan(temperatures[:,0])
    temperatures = temperatures[select]
    
    select = ~np.isnan(temperatures[:,1])
    temperatures = temperatures[select]
    
        
    for eq_temp in temperatures:
        J11 = (eq_temp[2]*mu1*I0*eq_temp[7] - 4*eq_temp[5]*sig*(eq_temp[0]**3)+(eq_temp[4]/(eq_temp[1] - eq_temp[0])**2))
        J22 = (eq_temp[3]*mu2*I0*eq_temp[7] - 4*eq_temp[5]*sig*(eq_temp[1]**3)+(eq_temp[4]/(eq_temp[0] - eq_temp[1])**2))
        J12 = -eq_temp[4]/((eq_temp[1] - eq_temp[0])**2)
        J21 = -eq_temp[4]/((eq_temp[0] - eq_temp[1])**2)
        stability = jacobian_stability_analysis(J11, J12, J21, J22)
        lambdas.append([stability, eq_temp[0], eq_temp[1], eq_temp[2], eq_temp[3], eq_temp[4], eq_temp[5], eq_temp[6], eq_temp[7]])

    
    
    lambdas = np.array(lambdas)
    for x in lambdas:
        print(' polar temp: ',x[1], '\n tropic temp: ', x[2], '\n polar constant, tropic constant : ', x[3], x[4], '\n Heat Flux constat: ', x[5], '\n emissivity: ', x[6], '\n a parameter: ', x[7], '\n b parameter: ', x[8], '\n', x[0], '\n')




    for x in lambdas:
        print(' polar temp: ',x[1], '\n tropic temp: ', x[2], '\n feebacks constant -a-: ', x[3], '\n feedbacks constant -b-: ',
               x[4], '\n Heat Flux Constant: ', x[5],'\n', x[0], '\n')

    np.save('stable_configuration', lambdas)
