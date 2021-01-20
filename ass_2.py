# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:53:20 2020

@author: Usuario
"""

#calculo derivatives

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

alpha=0.02

z,d= np.loadtxt("mars_soil.txt", unpack=True)
d=12600-d

error= 0.03*10**(4)*np.ones(len(d))
#I transform the points to a more useful basis so as to be able to fit them 
#easily


def gaussian_derivatives(A, c, z, f):
    #Calculates the derivatives necessary for computing the gradient
    GdA = 1/((2*np.pi)**(1/2)*c)*np.e**(-((z-f)**2)/(2*c**2))
    Gdc = A/((2*np.pi)**(1/2)*c)*np.e**(-((z-f)**2)/(2*c**2))*((z-f)**2/c**3-1/c)
    Gdf = A/((2*np.pi)**(1/2)*c)*np.e**(-((z-f)**2)/(2*c**2))*(+2*(z-f)/(2*c**2))
    return([GdA, Gdc, Gdf])

def lorentzian_derivatives(A,c,z,f):
    LdA = c**2*((z-f)**2+c**2)**(-1)
    Ldc = 2*c*A*((z-f)**2+c**2)**(-1) - A*c**2*((z-f)**2+c**2)**(-2)*2*c
    Ldf = +A*c**2*((z-f)**2+c**2)**(-2)*2*(z-f)
    return(LdA,Ldc,Ldf) 

def Tikinov_reg(G,d,epsilon= 0.7*10**(-6)):
    GGT=G.T@G
    m= np.linalg.inv(GGT + epsilon**2*np.identity(np.shape(GGT)[0]))@G.T@d
    return(m)

def gaussian(A, c, z, f):
    return(A/((2*np.pi)**(1/2)*c)*np.e**(-((z-f)**2)/(2*c**2)))

def lorentzian(A,c,z,f):
    return(A*c**2*((z-f)**2+c**2)**(-1))


#Initial parameters (totally rough estimation)
A=2500*np.ones(21)
c=0.25*np.ones(21)
f=[-10,-9,-8,-7 ,-6.5,-6, -5.5,-4, -3.5,-3, -1.5,1.5,3,3.5,4,6,6.5,7,8,9,10]

#gaussian: iter=150; alpha= 0.02
#lorentzian


#parameters of the while statement
it=0
itmax=2000
tolerance=5*10**(0)
chi2=[]
dm=[1E7]

d0= [] 
for i in range(len(z)):
    d0.append(sum(lorentzian(A[:],c[:],z[i], f[:]))) 


plt.figure()
plt.plot(z,-np.array(d0)+12600 , label="initial guess")

while (max(np.abs(dm))>=tolerance and it<=itmax):
    #I end my loop when the step on m is less t han the tolerannce
    G=np.zeros([len(z),len(A)*3])
    #reinizialize my G matrix that I will compute at each step
    for i in range(len(z)):
        aux=[]
        for j in range(len(A)):
            aux+=gaussian_derivatives(A[j], c[j], z[i], f[j])[:] 
            #I construct my matrix of derivatives
        G[i,:]=aux

    d0= [] 
    for i in range(len(z)):
        d0.append(sum(gaussian(A[:],c[:],z[i], f[:]))) 
        #i evaluate my function at my model and calculate the difference with
        # the data
    
    dd=d-d0
    chi2.append(np.linalg.norm(dd))
    dm=Tikinov_reg(G,dd, 0) #i use the newton method to calculate the step on the 
    #model 
    for j in range(len(A)):
        A[j]+=alpha*dm[3*j]; c[j]+=alpha*dm[3*j+1]; f[j]+=alpha*dm[3*j+2]    
    
    it+=1
    
#    if(it==60 or it==100 or it==125):
 #       alpha*=0.5

    


d0= []
for i in range(len(z)):
    
    d0.append(sum(gaussian(A[:],c[:],z[i], f[:])))

d= -d+12600 # i put then in the original basis
d0= -np.array(d0)+12600 


plt.plot(z,d, 'k.', markersize=2,label="data") 
plt.errorbar(z, d, yerr=error , fmt = 'k.', capsize=3, alpha=.2)

plt.plot(z,d0, label="optimal fit")
plt.xlabel("velocity (mm/s)")
plt.ylabel("Counts")
plt.legend(loc='lower left', fontsize=10)

plt.show()
plt.figure()
plt.plot(chi2)
plt.xlabel("iterations")
plt.ylabel("$|| d- m || $")

plt.show()

df= pd.DataFrame({"position":np.round(f,2),"width":np.round(c,2), "amplitude":np.round(A,2)})
print(df.to_latex(index=False, escape=False))

print("misfit:", chi2[-1])
print("number of steps:" , len(chi2) )
