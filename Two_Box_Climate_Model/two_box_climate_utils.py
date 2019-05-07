#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""This is a tool box for the Two Box Climate Scheme modlue. In this tool box are functions used for calculating
equilibrium position, its stability, and other quantities associated with the two box model created in the 2018-2019 
work of Sura & Sweeney"""


#BEGIN CODE


def linear_equilibrium(mu1, mu2, complex_guess_1, complex_guess_2):
    """The linear equilibrium function finds a two dimensional vector solution. The solutions we are looking for are real.
    Many solutions that can be found are complex and because of the numerical methods in which python finds these solutions
    we have decided to use the mpmath modules. This method of finding the solution for our non linear heat flux uses
    the muller method of numerical calculation. For more information on mpmath root finding: 
    http://mpmath.org/doc/1.1.0/calculus/optimization.html"""
    import numpy as np
    import sympy
    import mpmath
    #List of Constants
    I0 = 1367.0 #Solar constant W/m^2
    a = 2.8 #Unitless parameter needed for linear albedo feedback relationships more info in Budyko 1969
    b=.009 #Another parameter needed for linear feedback relationship more info in Budyko 1969
    sig = 5.67*10**-8 #Stephan boltzmann constant m^2 kg s^-2 K^-1
    e = .64 #Emmisivity of earth
    A = 2.7 #Heat flux parameter for non linear heat forcing
    solution_array = []
    for x in mu1:
        for y in mu2:
            #Below we find the two-dimensional vector solutions in form of [polar_solution, tropical_solution]
            f = [lambda T1,T2: (.156*x*I0*(1-a) + .156*x*I0*b*T1 - e*sig*(T1**4)+(A*(T2 - T1))), 
                 lambda T1,T2: (.288*y*I0*(1-a) + .288*y*I0*b*T2 - e*sig*(T2**4)+(A*(T1 - T2)))]
            solution = np.array(mpmath.findroot(f, ([complex_guess_1, complex_guess_2]), 
                                                solver='muller', verify = False))
            if abs(mpmath.im(solution[0])) > 1e-10:
                solution[0] = np.nan
            if abs(mpmath.im(solution[1])) > 1e-10:
                solution[1] = np.nan
            solution_array.append([mpmath.re(solution[0]), mpmath.re(solution[1]), x, y])
    return(solution_array)



def nonlinear_equilibrium(mu1, mu2, complex_guess_1, complex_guess_2):
    """The nonlinear equilibrium function finds a two dimensional vector solution. The solutions we are looking for are real.
    Many solutions that can be found are complex and because of the numerical methods in which python finds these solutions
    we have decided to use the mpmath modules. This method of finding the solution for our non linear heat flux uses
    the muller method of numerical calculation. For more information on mpmath root finding: 
    http://mpmath.org/doc/1.1.0/calculus/optimization.html"""
    import numpy as np
    import sympy
    import mpmath
    #List of Constants
    I0 = 1367.0 #Solar constant W/m^2
    a = 2.8 #Unitless parameter needed for linear albedo feedback relationships more info in Budyko 1969
    b=.009 #Another parameter needed for linear feedback relationship more info in Budyko 1969
    sig = 5.67*10**-8 #Stephan boltzmann constant m^2 kg s^-2 K^-1
    e = .64 #Emmisivity of earth
    A = 600 #Heat flux parameter for non linear heat forcing
    solution_array = []
    for x in mu1:
        for y in mu2:
            #Below we find the two-dimensional vector solutions in form of [polar_solution, tropical_solution]
            f = [lambda T1,T2: (.156*x*I0*(1-a) + .156*x*I0*b*T1 - e*sig*(T1**4)+(A/(T2 -T1))), 
                 lambda T1,T2: (.288*y*I0*(1-a) + .288*y*I0*b*T2 - e*sig*(T2**4)+(A/(T1 -T2)))]
            solution = np.array(mpmath.findroot(f, ([complex_guess_1, complex_guess_2]), 
                                                solver='muller', verify = False))
            if abs(mpmath.im(solution[0])) > 1e-10:
                solution[0] = np.nan
            if abs(mpmath.im(solution[1])) > 1e-10:
                solution[1] = np.nan
            solution_array.append([mpmath.re(solution[0]), mpmath.re(solution[1]), x, y])
    return(solution_array)


def non_linear_albedo_solver(mu1, mu2, guess_1, guess_2):
    """This solution is a modeling of a situation where solar intensity is decreased sufficiently until the earth 
    produces enough ice that the earth will reflect sufficient radiation to keep the earth in an ice age.
    Ice ages are still subject to a heat flux, the below function finds the equilibria for the ice ages for 
    made with a non-linear heat flux."""
    import numpy as np
    import sympy
    import mpmath
    #List of Constants
    I0 = 1367.0 #Solar constant W/m^2
    a = 2.8 #Unitless parameter needed for linear albedo feedback relationships more info in Budyko 1969
    b=.009 #Another parameter needed for linear feedback relationship more info in Budyko 1969
    sig = 5.67*10**-8 #Stephan boltzmann constant m^2 kg s^-2 K^-1
    e = .64 #Emmisivity of earth
    A = 600 #Heat flux parameter for non linear heat forcing
    solution_array = []
    for x in mu1:
        for y in mu2:
            #Below we find the two-dimensional vector solutions in form of [polar_solution, tropical_solution]
            f = [lambda T1,T2: (.156*x*I0*(1-.75) - e*sig*(T1**4)+(A/(T2 -T1))), 
                 lambda T1,T2: (.288*y*I0*(1-.75) - e*sig*(T2**4)+(A/(T1 -T2)))]
            solution = np.array(mpmath.findroot(f, ([complex_guess_1, complex_guess_2]), 
                                                solver='muller', verify = False))
            if abs(mpmath.im(solution[0])) > 1e-10:
                solution[0] = np.nan
            if abs(mpmath.im(solution[1])) > 1e-10:
                solution[1] = np.nan
            solution_array.append([mpmath.re(solution[0]), mpmath.re(solution[1]), x, y])
    return(solution_array)

def linear_albedo_solver(mu1, mu2, guess_1, guess_2):
    """For the case of unstable solutions and ranges of the plot, we must have a stable linear albedo feeback solution.
    This solution is a modeling of a situation where solar intensity is decreased sufficiently until the earth 
    produces enough ice that the earth will reflect sufficient radiation to keep the earth in an ice age.
    Finding equilibria for the ice age solutions using a linear heat flux."""
    import numpy as np
    import sympy
    import mpmath
    #List of Constants
    I0 = 1367.0 #Solar constant W/m^2
    a = 2.8 #Unitless parameter needed for linear albedo feedback relationships more info in Budyko 1969
    b=.009 #Another parameter needed for linear feedback relationship more info in Budyko 1969
    sig = 5.67*10**-8 #Stephan boltzmann constant m^2 kg s^-2 K^-1
    e = .64 #Emmisivity of earth
    A = 3 #Heat flux parameter for linear heat forcing
    solution_array = []
    for x in mu1:
        for y in mu2:
            #Below we find the two-dimensional vector solutions in form of [polar_solution, tropical_solution]
            f = [lambda T1,T2: (.156*x*I0*(1-.75) - e*sig*(T1**4)+(A*(T2 -T1))), 
                 lambda T1,T2: (.288*y*I0*(1-.75) - e*sig*(T2**4)+(A*(T1 -T2)))]
            solution = np.array(mpmath.findroot(f, ([complex_guess_1, complex_guess_2]), 
                                                solver='muller', verify = False))
            if abs(mpmath.im(solution[0])) > 1e-10:
                solution[0] = np.nan
            if abs(mpmath.im(solution[1])) > 1e-10:
                solution[1] = np.nan
            solution_array.append([mpmath.re(solution[0]), mpmath.re(solution[1]), x, y])
    return(solution_array)
    
def non_linear_stability_analysis(branch):
    """The structure of the bifurcation fold leads to the question of stability for each of the braches. 
    In order to analyze the stability of each of these branches we use the technique of finding stabilty used
    in the Fraedrich 1978 paper. This is a linearization method where each of the coupled differential 
    equations are differentiated and put into a jacobian matrix and eigenvalues are found. If eigenvalues are
    both positive the equilibrium position is an unstable node, if both negative a stable node, if one postive 
    and one negative it is unstable saddlepoint, if we have nonreal solutions then there can be no certainty in 
    the stability."""
    import numpy as np
    import mpmath    
    #List of Constants
    I0 = 1367.0 # Solar constant W/m^2
    b=.009 # Parameter needed for linear feedback relationship more info in Budyko 1969 K^(-1)
    sig = 5.67*10**-8 # Stephan boltzmann constant m^2 kg s^-2 K^-1
    e = .64 # Emmisivity of earth
    A = 600 # Heat flux parameter. Derived for mixing timescales of 8-16 months and temp difference 20-30 deg K
    select = ~np.isnan(branch[:,0])
    branch = branch[select]
    stability_verdicts = []
    for eq_position in branch:
        polartemp = eq_position[0]
        tropicalttemp = eq_position[1]
        pmu = eq_position[2] # Polar Mu Value: regulates solar intensity in polar box.
        tmu = eq_position[3] # Tropical Mu Value: regulates solar intensity in tropical box.
        J11 = (.156*pmu*I0*b - 4*e*sig*(polartemp**3) + A/((tropicaltemp - polartemp)**2)) #Jacobian position 11
        J22 = (.288*tmu*I0*b - 4*e*sig*(tropicaltemp**3) + A/((polartemp-tropicaltemp)**2))#Jacobian 22
        J12 = -A/((tropicaltemp - polartemp)**2)
        J21 = -A/((polartemp - tropicaltemp)**2)

        f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
        z1 = np.array(mpmath.findroot(f, ([150+1j]), solver='muller', verify = False))
        z2 = np.array(mpmath.findroot(f, ([-150+1j]), solver='muller', verify = False))
        stability_verdicts.append([z1,z2])
        

    return(stability_verdicts)


def stability_analysis(branch, linear_or_non_linear):
    """The structure of the bifurcation fold leads to the question of stability for each of the braches. 
    In order to analyze the stability of each of these branches we use the technique of finding stabilty used
    in the Fraedrich 1978 paper. This is a linearization method where each of the coupled differential 
    equations are differentiated and put into a jacobian matrix and eigenvalues are found. If eigenvalues are
    both positive the equilibrium position is an unstable node, if both negative a stable node, if one postive 
    and one negative it is unstable saddlepoint, if we have nonreal solutions then there can be no certainty in 
    the stability."""
    import numpy as np
    import mpmath    
    #List of Constants
    I0 = 1367.0 # Solar constant W/m^2
    b=.009 # Parameter needed for linear feedback relationship more info in Budyko 1969 K^(-1)
    sig = 5.67*10**-8 # Stephan boltzmann constant m^2 kg s^-2 K^-1
    e = .64 # Emmisivity of earth
    A = 600 # Heat flux parameter. Derived for mixing timescales of 8-16 months and temp difference 20-30 deg K
    select = ~np.isnan(branch[:,0])
    branch = branch[select]
    stability_verdicts = []
    
    if linear_or_non_linear == str('non-linear'):
        for eq_position in branch:
            polartemp = eq_position[0]
            tropicaltemp = eq_position[1]
            pmu = eq_position[2] # Polar Mu Value: regulates solar intensity in polar box.
            tmu = eq_position[3] # Tropical Mu Value: regulates solar intensity in tropical box.
            J11 = (.156*pmu*I0*b - 4*e*sig*(polartemp**3) + A/((tropicaltemp - polartemp)**2)) #Jacobian position 11
            J22 = (.288*tmu*I0*b - 4*e*sig*(tropicaltemp**3) + A/((polartemp-tropicaltemp)**2))#Jacobian 22
            J12 = -A/((tropicaltemp - polartemp)**2)
            J21 = -A/((polartemp - tropicaltemp)**2)

            f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
            z1 = np.array(mpmath.findroot(f, ([150+1j]), solver='muller', verify = False))
            z2 = np.array(mpmath.findroot(f, ([-150+1j]), solver='muller', verify = False))
            stability_verdicts.append([z1,z2])
    
    if linear_or_non_linear == str('linear'):
        for eq_position in branch:
            polartemp = eq_position[0]
            tropicaltemp = eq_position[1]
            pmu = eq_position[2] # Polar Mu Value: regulates solar intensity in polar box.
            tmu = eq_position[3] # Tropical Mu Value: regulates solar intensity in tropical box.
            J11 = (.156*pmu*I0*b - 4*e*sig*(polartemp**3) - A) #Jacobian position 11
            J22 = (.288*tmu*I0*b - 4*e*sig*(tropicaltemp**3) - A) #Jacobian 22
            J12 = A
            J21 = A

            f = [lambda z: ((J11 - z)*(J22 - z)- (J12)*(J21))]
            z1 = np.array(mpmath.findroot(f, ([150+1j]), solver='muller', verify = False))
            z2 = np.array(mpmath.findroot(f, ([-150+1j]), solver='muller', verify = False))
            stability_verdicts.append([z1,z2])

    return(stability_verdicts)


