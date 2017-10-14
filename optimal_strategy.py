#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:29:10 2017

@author: okadatomoki
"""

import numpy as np
from cost_function import *

def phi_ita(t):
    return 1/(2*(2+rho*(T-t))*(1+np.exp(-ita*(T-t))+nu*rho*(T-t)*zeta(ita*(T-t)) \ 
                 +beta/rho(2+rho*(T-t)*(1+zeta(ita*(T-t))+nu*rho*(T-t)*omega(ita*(T-t)))))

def Phi_0(s,t):
    first_term = (beta/rho+nu/2*(1/2-beta/rho))*(np.exp(-beta*s)-np.exp(-beta*t))/beta
    second_term = (1-nu)*(1-beta/rho)*np.exp(-beta*T)/rho*(L(rho,beta,T-s)-L(rho,beta,T-t))
    third_term = nu/4*((T-t)*np.exp(-beta*s))-(T-t)*np.exp(-beta*t)
    return first_term + second_term + third_term
    
def Phi_ita(s,t):
    first_term = 1/2*(1/rho+nu/ita)*(np.exp(-beta*s)-np.exp(-beta*t))
    second_term = np.exp(-beta*T)/(2*rho)*(1+nu*(rho-2*beta)/ita+beta/ita*(1-nu*rho/ita))*(L(rho,beta,T-s)-L(rho,beta,T-t))
    third_term = np.exp(-beta*T)/(2*rho)*(1-nu*rho/ita-beta/ita*(1-nu*rho/ita))*(L(rho,alpha,T-s)-L(rho,alpha,T-t))
    return first_term + second_term + third_term
    
def delta_t():
    return delta_0*np.exp(-beta*t)+np.exp(-beta*t)*Theta(chi_t)
    
def Theta(i):
    
def Delta_X_0_OW():
    return -x_0/(2*rho*T)

def Delta_X_T_OW():
    return -x_0/(2*rho*T)
    
def Delta_X_T_OW(n):
    dt = T/n
    return -rho*x_0*dt/(2+rho*T)
       
def Delta_X_trend_0():
    return (delta_0*m_1/2*rho*(2+rho*T*(1+zeta(ita*T)+nu*rho*T*omega(ita*T)))-(1+rho*T)*q*D_0)/((2+rho*T)*(1-epsilon))
    
def Delta_X_trend_T():
    first_term = delta_0*m_1/(2*rho)*((2+rho*T*(1+zeta(ita*T)+nu*rho*T*omega(ita*T)))/(2*rho*T)-2rho*Phi_ita(0,T)-2*np.exp(-beta*T))
    second_term = q*D_0/(2+rho*T)
    return (first_term + second_term)/(1 - epsilon)

def Delta_X_trend_t(n):
    dt = T/n
    first_term = delta_0*m_1/(2*rho)*((2+rho*T*(1+zeta(ita*T)+nu*rho*T*omega(ita*T)))/(2*rho*T)-2rho*Phi_ita(0,T)-2*phi_ita(t)*np.exp(-beta*T))*rho*dt 
    second_term = q*D_0/(2+rho*T)*rho*dt
    return (first_term + second_term)/(1 - epsilon)
    
def Delta_X_dyn_0():
    return 0
    
def Delta_X_dyn_T():
    return None
    
def Delta_X_dyn_t():
    return None
    
   