# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:27:19 2020

@author: zhang
"""
import numpy as np
import math


angle = np.sqrt(20.8*20.8 + 20.8*20.8)/180*np.pi
A_air = math.cos(angle/2)

N_star = 6.57*np.exp(1.08*3)*(1 - A_air)/2

P_N  = 1 - np.exp(-N_star)*N_star

print(P_N)