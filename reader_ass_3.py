# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:19:28 2020

@author: To+
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#empiezan cosas buenas

x,y= np.loadtxt("ass_3_data.txt",unpack=True)
y_err=1E-5*np.ones(len(y))

y*=1E-5

#constantes:

d_rho=-1700 #kg/m**3
G= 6.67E-11 #mgal m^2/kg^2
sigma_n= 300 # m
sigma_m= 1E-5# mgal 

mu=1 #yo esto te juro que no lo entiendo
a=3420 #m
l=100 # 20 #ISAAAC NO LO PUTO TOQUES (en principio con esto tenemos un acc_rate de 30-70%)
delta=0.005 #para evitar que pete


f= interp1d(x, y, fill_value="extrapolate")
xl=np.linspace(0,a, 100) #m


Deltagaux= f(xl) #pa probar (cambiar con los datos a priori DONE)
m0=Deltagaux/(2*np.pi*G*d_rho) #aproximación burda de ecuación (1)

M=len(m0)
delta_x=a/M

x_base=[]
for i in range(M-1):
    x_base.append(i*delta_x)

x_base=np.array(x_base)
def delta_g_j(x_j,h):

    """
    Parameters
    ----------
    x_j : The location of the input detector (float)
    h : The discrete distribution of ice thicknesses. (array)
    Returns
    -------
    Delta_g at x_j (float) computed by discretizing the integral of equation
    (1) in the assignment

    """
    dg=0
    for l in range(M-1):
        x_base_l = x_base[l]
        x_top_l = x_base[l]+ delta_x
        h_l=h[l]
        dg+= G*d_rho*( x_top_l*np.log((h_l**2+(x_j-x_top_l)**2)/((x_j-x_top_l)**2+delta))+x_j*np.log(delta+(x_j-x_top_l)**2) + 2*np.sqrt(delta)*np.arctan((x_j-x_top_l)/np.sqrt(delta))-x_j*np.log(h_l**2+(x_j-x_top_l)**2)-2*h_l*np.arctan((x_j-x_top_l)/h_l))
        dg-= G*d_rho*( x_base_l*np.log((h_l**2+(x_j-x_base_l)**2)/((x_j-x_base_l)**2+delta))+x_j*np.log(delta+(x_j-x_base_l)**2) + 2*np.sqrt(delta)*np.arctan((x_j-x_base_l)/np.sqrt(delta))-x_j*np.log(h_l**2+(x_j-x_base_l)**2)-2*h_l*np.arctan((x_j-x_base_l)/h_l))                
    return(dg)


#acaban cosas  buenas



s=np.loadtxt("results_m.txt")
L_m=np.loadtxt("L_m.txt")

L_m2=L_m[2001:]


  
        
k=s.reshape((int(len(s)/100), 100))  


"""
k=[]
for i in range(len(L_m2)):
    if(L_m2[i]>1E-9):
        k.append(list(k2[i,:]))
k=np.array(k)
"""
s2=[]
s2std=[]
s3=[]
s3std=[]

for i in range(100):
    s2.append(np.mean(k[:,i]))
    s2std.append(np.std(k[:,i]))
    
for i in range(len(x)):
    s3aux=[]
    for j in range(100):
        s3aux.append(delta_g_j(x[i],k[j,:]))
    s3.append(s3aux)    
s3=np.array(s3)

s4=[]
s4err=[]
for i in range(np.shape(s3)[0]):
    s4.append(np.mean(s3[i,:]))    
    s4err.append(np.std(s3[i,:]))    
xl=np.linspace(0,3420, len(s2))    


#perfil de alturas de hielo
plt.figure()    
plt.plot(xl,s2, 'b.')  
plt.errorbar(xl, s2, yerr=s2std/np.sqrt(80000/100), fmt = 'b.', capsize=3, alpha=.5)
plt.xlabel("x (m) ")
plt.ylabel("h(x) (m)")


#monitor de log(L)
plt.figure()
plt.plot(L_m)
plt.xlabel("iterations")
plt.yscale("log")
plt.ylabel("L(m)")

plt.figure()
plt.plot(x,delta_g_j(x,m0), 'g.', label="initial model")
plt.plot(x,s4, 'b.')
plt.errorbar(x, s4, yerr=s4err/np.sqrt(80000/100) , fmt = 'r.', capsize=3, alpha=.5, label="final model ")
#plt.plot(x,y)
plt.errorbar(x, y, yerr=y_err , fmt = 'b.', capsize=3, alpha=.5, label="data")

plt.xlabel("m")
plt.ylabel("$m/s^{2} $")
plt.legend(fontsize=9, loc="best")

"""

#ALICIA PORFA HAZ ALGO QUE NO PUEDO MAS CON ESTA MIERDA :'(

fig, axs = plt.subplots(10,3)
for i in range(10):
    for j in range(3):
        axs[i, j].hist(k[:, i*10+j], 30)
        axs[i,j].set_title(str( i*10+j))

fig2, axs2 = plt.subplots(10,2)        
for i in range(10):
    for j in range(2):
        axs2[i, j].hist(k[:, i*10+j+3], 30)
        axs2[i,j].set_title(str(i*10+j+3))
        
fig3, axs3 = plt.subplots(10,2)
for i in range(10):
    for j in range(2):
        axs3[i, j].hist(k[:, i*10+(j+5)], 30)
        axs3[i,j].set_title(str( i*10+(j+5)))
        
fig4, axs4 = plt.subplots(10,3)
for i in range(10):
    for j in range(3):
        axs4[i, j].hist(k[:, i*10+(j+7)], 30)   
        axs4[i,j].set_title(str( i*10+(j+7)))

"""
#EJERCICIO 9

x = k[:, 51]
y = k[:, 52]

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(figsize=(8, 8))

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)

# the scatter plot:
ax_scatter.scatter(x, y)

# now determine nice limits by hand:
binwidth = 0.25
lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
ax_scatter.set_xlim((-lim, lim))
ax_scatter.set_ylim((-lim, lim))

bins = np.arange(-lim, lim + binwidth, binwidth)
ax_histx.hist(x, bins=bins)
ax_histy.hist(y, bins=bins, orientation='horizontal')

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())

plt.show()