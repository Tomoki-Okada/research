#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:10:03 2017

@author: okadatomoki
"""

import numpy as np
from scipy.integrate import quad

def parameter_set():
    nu
    epsilon
    rho
    _lambda
    r
    iota
    kappa_inf
    m_1
    m_2
    alpha_tilda
    alpha_2
    beta

def zeta(y):
    if y == 0:
        return 1
    else:
        return (1-np.exp(y))/y

def omega(y):
    if y == 0:
        return 1
    else:
        return (np.exp(-y)-zeta(y))/y

def omega_prime(y):
    return (2*zeta(y)-1-np.exp(-y))/(y**2)

def calG_ita(u):
    return zeta(ita*u)+nu+rho*u*omega(ita*u)

def c_hat_ita(u):
    return 1/(1-epsilon)*(ita-nu*rho)**2*(rho*u**3)/8*omega_prime(ita*u)*zeta(ita*u)
    
def L(r, _lambda, t):
    return np.exp(-2*_lambda/r)*(calE(_lambda/r*(2+r*t))-calE(2*_lambda/r))
    
def calE(y):
    def h(u):
        return np.exp(-u)/u  
    return -quad(h, a = -y, b = np.inf)[0]
    
def calI(p, u):
    def h(s):
        return s**p*np.exp(-2*iota_c*s)
    return np.exp(2*iota_c*u)*quad(h, a = 0, b = u)[0]
    
def e(u):
    first_term = (-(1-nu)**2)/(1-epsilon)*(m_2-(m_1*(2*alpha_tilda*rho-alpha_2*m_1))/(rho**2)) \
                *(calI(0,u)/2-np.exp(2*iota_c*u)/rho*L(rho,-2*iota_c,u))
    second_term = (nu*(1-nu)*m_1)/(2*rho**2*(1-epsilon))*(alpha_tilda-(alpha_2*m_1)/rho)*rho**2*calI(1,u)
    third_term = (alpha_2*nu**2*m_1**2)/(4*rho**3*(1-epsilon))*(rho**2*calI(1,u)+1/2*rho**3*calI(2,u)+1/12*rho**4*calI(3,u))
    return first_term + second_term + third_term
    
def g(u):
    first_term = -2*beta*kappa_inf*(1-nu)**2/(1-epsilon)*(m_2-m_1*(2*alpha_tilda*rho- alpha_2+m_1)/(rho**2)) \
                *(calI(1,u)/2-1/(2*iota_c*rho)*(np.exp(2*iota_c*u)*L(rho,-2*iota_c,u)-np.log(1+rho*u/2)))
    second_term = (beta*kappa_inf*nu*(1-nu)*m_1)/(2*rho**3*(1-epsilon))*(alpha_tilda-alpha_2*m_1/rho)*rho**3*calI(2,u)
    third_term = (beta*kappa_inf*alpha_2*nu**2*m_1**2)/(4*rho**4*(1-epsilon))*(rho**3*calI(2,u)+1/3*rho**4*calI(3,u)+1/24*rho**5*calI(4,u))
    return first_term + second_term + third_term
    
def Cost(q, t, x, d, z, delta, Sigma):
    """
    input:
        q
        t
        x
        d
        z
        delta
        Sigma
    """
    first_block = -q(z+d)*x+((1-epsilon)/(2+rho*(T-t))+epsilon/2)*x**2
    second_block = -1/(1-epsilon)*(rho*(T-t)/2)/(2+rho*(T-t))*(q*d-calG_ita(T-t)*delta*m_1/rho)*x
    third_block = -1/(1-epsilon)*(rho*(T-t)/2)/(2+rho*(T-t))*(q*d-calG_ita(T-t)*delta*m_1/rho)**2
    forth_block = c_hat_ita(T-t)*(delta*m_1/rho)**2+e(T-t)*Sigma+g(T-t)
    return (first_block + second_block + third_block + forth_block)/q    
    