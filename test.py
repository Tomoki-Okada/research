#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:46:46 2017

@author: okadatomoki
"""

import matlab.engine

eng = matlab.engine.start_matlab()
ret = eng.test(1.0,5.0)
print(ret)