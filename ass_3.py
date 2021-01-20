# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:28:55 2020

@author: To+
"""


from numba import njit

import numpy as np
import numpy.random as rand
from scipy.interpolate import interp1d

x,y= np.loadtxt("ass_3_data.txt",unpack=True)
y *= 1E-5 #1 mgal is 1e-5 m/s^2

#Constants:
d_rho = -1700 #kg/m**3
G = 6.67E-11 #m^3/(kg s^2)
sigma_n = 300 #m
sigma_m = 1E-5 #m/s^2

mu = 1 
a = 3420 #m
l = 100 # 20 #ISAAAC NO LO PUTO TOQUES (with this we have an acc_rate of 30-70%)
delta = 0.005 #m 

f = interp1d(x, y, fill_value="extrapolate")
xl = np.linspace(0,a, 100) #m


Deltagaux = f(xl) 
m0 = Deltagaux/(2*np.pi*G*d_rho) #approximation equation (1)

M = len(m0)
delta_x = a/M

x_base = []
for i in range(M-1):
    x_base.append(i*delta_x)
x_base = np.array(x_base)
m = m0.copy(); m += rand.rand(len(m))*l-l

@njit
def delta_g_j(x_j,h):
    """
    
    Parameters
    ----------
    x_j : The location of the input detector (float)
    h   : The discrete distribution of ice thicknesses (array)
    
    Returns
    -------
    Delta_g at x_j (float) computed by discretizing the integral of equation
    (1) in the assignment

    """
    def integral_discret(x,h_l):
        return x*np.log((h_l**2+(x_j-x)**2)/((x_j-x)**2+delta)) + x_j*np.log(delta+(x_j-x)**2) + 2*np.sqrt(delta)*np.arctan((x_j-x)/np.sqrt(delta)) - x_j*np.log(h_l**2+(x_j-x)**2) - 2*h_l*np.arctan((x_j-x)/h_l)
    dg=0
    for l in range(M-1):
        x_base_l = x_base[l]
        x_top_l = x_base[l] + delta_x
        h_l = h[l]
        dg += G*d_rho*integral_discret(x_top_l,h_l)
        dg -= G*d_rho*integral_discret(x_base_l,h_l)
    return(dg)


C_m = sigma_m**(2)*np.identity(len(y))

@njit(parallel=True)
def L(m):
    """

    Parameters
    ----------
    m : Our model (array)

    Returns
    -------
    The likelihood function L. When the residual (d_obs-g(m)) ->0, L->1 and
    when (d_obs-g(m)) -> inf, L->0. 

    """
    g = []
    for j in range (len(x)):
        g.append(delta_g_j(x[j], m))
    g = np.array(g)
    r = y-g

    L = np.e**(-1/2*r.T@np.linalg.inv(C_m)@r)
   
    return(L)

C_rho = sigma_n**(2)*np.identity(M) 

@njit(parallel=True)
def rho(m):
    """
    
    Parameters
    ----------
    m : Our model (array)

    Returns
    -------
    The a priori probability density rho. When the residual (m-m0) ->0, rho->1
    and when (m-m0) -> inf, rho->0. 

    """
    r = m-m0
 
    rho=1*np.e**(-1/2*r.T@np.linalg.inv(C_rho)@r ) 
    return(rho)

@njit()
def sigma(m):
    """

    Parameters
    ----------
    m : model parameters (array)
    
    Returns
    -------
    full probability distribution from which the MCMC step of the chain 
    is computed

    """
    return (L(m)*rho(m)) 


it=0
itmax=10000

a1=0; a2=0; a3=0

#MCMC evolution
cont=0
monitor=[]

while(it<itmax):
    j = rand.randint(0, len(m))
    maux = m.copy()
    maux[j] += l*(2*rand.rand()-1)
    
    
    L_m = L(m)
    monitor.append(L_m)
    
    acc = sigma(maux)/(L_m*rho(m))
    if (acc>1):
        m = maux.copy();
        cont += 1
    elif (rand.rand()<acc):
        m = maux.copy()
        cont += 1
    if (it%2000==0):
        print(monitor[-1])
        print(it)
     
    if (L_m>1E-6 and a1==0):
        a1 = 1
        l = 400
    
    #save data in equilibrium
    if (it>2000): 
       
        with open("results_m.txt", "ab") as f:
            np.savetxt(f, m)
    it += 1

np.savetxt("L_m.txt",monitor)