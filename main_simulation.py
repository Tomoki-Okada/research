#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:12:04 2017

@author: okadatomoki
"""

import matlab.engine

eng = matlab.engine.start_matlab()
mu1 = 1
mu2 = 2
a11 = 0.1
a12 = 0.3
a21 = 0.3
a22 = 0.1
w = 0.5
ret = eng.Hawkes_Simulation(mu1, mu1, a11, a12, a21, a22, w)