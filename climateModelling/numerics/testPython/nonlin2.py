#non linear pendulum 

# code for nonlin and lin theta variation with time taken from https://www.youtube.com/watch?v=_eZyTNthJG4 
# Potential Energy and Kinetic Energy added.

import numpy as np
from  numpy import sin, cos
from scipy.integrate import odeint
from matplotlib import pyplot as plt

import math


#define equation
def equations(y0, t):
    theta, x = y0
    f = [x, -(g/l) * sin(theta)]
    return f

def plot_results(time, theta1, theta2, PE_lin, PE_non_lin, KE_lin, KE_non_lin):
    plt.figure(1)
    plt.plot(time, theta1[:,0])
    plt.plot(time, theta2)

    s = '(Initial Angle = ' + str(initial_angle) + ' degrees)'
    plt.title('Pendulum Motion: ' + s)
    plt.xlabel('time (s)')
    plt.ylabel('angle (rad)')
    plt.grid(True)
    plt.legend(['nonlinear', 'linear'], loc='lower right')
    plt.show()

    plt.figure(2)
    plt.plot(time, PE_non_lin)
    plt.plot(time, PE_lin)

    s = '(Initial Angle = ' + str(initial_angle) + ' degrees)'
    plt.title('Pendulum Potential Energy: ' + s)
    plt.xlabel('time (s)')
    plt.ylabel('PE, Joule')
    plt.legend(['nonlinear', 'linear'], loc='lower right')
    plt.show()

    plt.figure(3)
    plt.plot(time, KE_non_lin)
    plt.plot(time, KE_lin)

    s = '(Initial Angle = ' + str(initial_angle) + ' degrees)'
    plt.title('Pendulum Kinetic Energy: ' + s)
    plt.xlabel('time (s)')
    plt.ylabel('KE, Joule')
    plt.legend(['nonlinear', 'linear'], loc='lower right')
    plt.show()

#setup paramters
g = 9.81
l = 1.0
m = 2
time = np.arange(0, 20.0, 0.01)

#initial condition
initial_angle = 90
theta0 = np.radians(initial_angle)
x0 = np.radians(0.0)

#find soulution to the non linear problem
theta1 = odeint(equations, [theta0, x0], time)

#find solution to linear problem
w = np.sqrt(g/l)
theta2 = [theta0 * cos(w*t) for t in time]



####### POTENTIAL ENERGY
# PE = m*g*h  (mass * grav acc * height)
# h = l-lcos(theta)


#define equation
def equation_PE(theta, m, g):
    h = np.zeros(len(theta))
    h = h + l
    h = h - l*cos(theta)
    f2 = m * g * h
    return f2

 
print theta1[0] 
print theta2[0]  

PE_non_lin = equation_PE(theta1[:,0], m, g)
PE_lin = equation_PE(np.array(theta2), m, g)


###### KINETIC ENERGY
#KE = 1/2 * m * v^2
#v = (2g * L * (1-cos(a)))^-1/2

#define equation velocity
def equation_v(theta, g, l):
    v = np.zeros(len(theta))
    f3 = np.sqrt(2 * g * l * (1-cos(theta)))
    return f3

#calculate velocity
V_non_lin = equation_v(theta1[:,0], g, l)
V_lin = equation_v(np.array(theta2), g, l)

#find kinetic energy
def equation_KE(theta, m, g, l):
    KE = np.zeros(len(theta))
    v = equation_v(theta, g, l)
    f4 = 0.5 * m * np.power(v, 2)
    return f4

#calculate KE
KE_non_lin = equation_KE(V_non_lin, m, g, l)
KE_lin = equation_KE(V_lin, m, g, l)


#plot results
plot_results(time, theta1, theta2, PE_lin, PE_non_lin, KE_lin, KE_non_lin)




