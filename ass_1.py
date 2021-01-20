# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:21:39 2020

@author: Usuario
"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import pandas as pd

def noise(t):
    n=[]
    for i in range(len(tgammal)):
        n.append(rand.gauss(0, 1))
    n=np.array(n)
    n*=np.linalg.norm(tgammar)/(20*np.linalg.norm(n))
    return(n)

def Tikinov_reg(G,d,epsilon=0.7*10**(-6)):
    GGT=G.T@G
    m= np.linalg.inv(GGT + epsilon**2*np.identity(np.shape(GGT)[0]))@G.T@d
    return(m)


x1= 4; x2= 7; y1= -1; y2= -9

xs=[2,3,4,5,6,7,8,9,10,11] 
xmax=13

s=1
tgammar=[]
tgammal=[]

for x0 in xs:
    n=0
    u=np.sqrt(2)*(xmax-x0)
    cont=0
    while n<100000.:
        du=u/np.sqrt(2)*rand.random()
        x= x0+du; y=-du
        if(x>=x1 and x<=x2 and y<=y1 and y>=y2):
            cont+=1.
        n+=1. 
    tgammar.append(2*u*cont/n)  

for x0 in xs:
    n=0
    u=np.sqrt(2)*(x0)
    cont=0
    while n<100000.:
        du=u/np.sqrt(2)*rand.random()
        x= x0-du; y=-du
        if(x>=x1 and x<=x2 and y<=y1 and y>=y2):
            cont+=1.
        n+=1. 
    tgammal.append(2*u*cont/n)  

 
tgammar=np.array(tgammar)
tgammal=np.array(tgammal)

tgammar+=noise(tgammar)
tgammal+=noise(tgammal)

df= pd.DataFrame({"Detector":[1,2,3,4,5,6,7,8,9,10] ,"right":tgammar, "left":tgammal})
print(df.to_latex(index=False, escape=False))

#print(tgammar+n, tgammal)


G=np.zeros([20, 13*11])

for i in range(10):
    k=i+2
    for j in range(11-i):
        G[i, k]=2*np.sqrt(2)
        k+= 1+13



for i in range(10,20):
    k=i+1-10
    for j in range(i+2-10):
        G[i, k]= 2*np.sqrt(2) 
        k+= 1+11

d=np.concatenate((tgammar, tgammal))

m=Tikinov_reg(G,d)
plt.imshow(np.reshape(m,[11,13]))
plt.colorbar()

