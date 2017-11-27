#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:40:03 2017

@author: xinruyue
"""
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt


beta = 1.0
gamme = 0.14

TS = 1.0
S0 = 1 - 1e-6
I0 = 1e-6

INPUT = [S0, I0, 0.0]
t = np.arange(0,71,1)

def pend(INP, t):
    Y = np.zeros(3)
    V = INP
    Y[0] = -beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamme * V[1]
    Y[2] = gamme * V[1]

    return Y

res = spi.odeint(pend, INPUT, t)

plt.figure()
plt.plot(res[:,0], label = 'S')
plt.plot(res[:,1], label = 'I')
plt.plot(res[:,2], label = 'R')
plt.legend(loc = 'best')
plt.xlabel('t')
plt.grid()
plt.show()