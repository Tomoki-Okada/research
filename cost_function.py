#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:10:03 2017

@author: okadatomoki
"""

import numpy as np
from scipy.integrate import quad

class Cost():

    def set_up(self, data):
        """
        パラメータのセットアップ
        """
        self.q = data['q']
        self.T = data['T']
        self.nu = data['nu']
        self.epsilon = data['epsilon']
        self.rho = data['rho']
        self.iota_c = data['iota_c']
        self.kappa_inf = data['kappa_inf']
        self.ita = data['ita']
        self.m_1 = data['m_1']
        self.m_2 = data['m_2']
        self.alpha_tilda = data['alpha_tilda']
        self.alpha_2 = data['alpha_2']
        self.beta = data['beta']
    
    def zeta(self, y):
        if y == 0:
            return 1
        else:
            return (1-np.exp(y))/y
    
    def omega(self, y):
        if y == 0:
            return 1
        else:
            return (np.exp(-y)-self.zeta(y))/y
    
    def omega_prime(self, y):
        if y == 0:
            return -1/6
        else:
            return (2*self.zeta(y)-1-np.exp(-y))/(y**2)
    
    def calG_ita(self, u):
        return self.zeta(self.ita*u)+self.nu+self.rho*u*self.omega(self.ita*u)
    
    def c_hat_ita(self, u):
        return 1/(1-self.epsilon)*(self.ita-self.nu*self.rho)**2*(self.rho*u**3) \
                /8*self.omega_prime(self.ita*u)*self.zeta(self.ita*u)
        
    def L(self, r, _lambda, t):
        return np.exp(-2*_lambda/r)*(self.calE(_lambda/r*(2+r*t))-self.calE(2*_lambda/r))
        
    def calE(self, y):
        def h(u):
            return np.exp(-u)/u  
        return -quad(h, a = -y, b = np.inf)[0]
        
    def calI(self, p, u):
        def h(s):
            return s**p*np.exp(-2*self.iota_c*s)
        return np.exp(2*self.iota_c*u)*quad(h, a = 0, b = u)[0]
        
    def e(self, u):
        first_term = (-(1-self.nu)**2)/(1-self.epsilon)*(self.m_2-(self.m_1 \
                    *(2*self.alpha_tilda*self.rho-self.alpha_2*self.m_1))/(self.rho**2)) \
                    *(self.calI(0,u)/2-np.exp(2*self.iota_c*u)/self.rho*self.L(self.rho,-2*self.iota_c,u))
        second_term = (self.nu*(1-self.nu)*self.m_1)/(2*self.rho**2*(1-self.epsilon)) \
                    *(self.alpha_tilda-(self.alpha_2*self.m_1)/self.rho)*self.rho**2*self.calI(1,u)
        third_term = (self.alpha_2*self.nu**2*self.m_1**2)/(4*self.rho**3*(1-self.epsilon)) \
                    *(self.rho**2*self.calI(1,u)+1/2*self.rho**3*self.calI(2,u)+1/12*self.rho**4*self.calI(3,u))
        return first_term + second_term + third_term
        
    def g(self, u):
        first_term = -2*self.beta*self.kappa_inf*(1-self.nu)**2/(1-self.epsilon) \
                    *(self.m_2-self.m_1*(2*self.alpha_tilda*self.rho- self.alpha_2+self.m_1)/(self.rho**2)) \
                    *(self.calI(1,u)/2-1/(2*self.iota_c*self.rho)*(np.exp(2*self.iota_c*u)*self.L(self.rho,-2*self.iota_c,u)-np.log(1+self.rho*u/2)))
        second_term = (self.beta*self.kappa_inf*self.nu*(1-self.nu)*self.m_1)/(2*self.rho**3*(1-self.epsilon)) \
                    *(self.alpha_tilda-self.alpha_2*self.m_1/self.rho)*self.rho**3*self.calI(2,u)
        third_term = (self.beta*self.kappa_inf*self.alpha_2*self.nu**2*self.m_1**2) \
                    /(4*self.rho**4*(1-self.epsilon))*(self.rho**3*self.calI(2,u)+1/3*self.rho**4*self.calI(3,u)+1/24*self.rho**5*self.calI(4,u))
        return first_term + second_term + third_term
        
    def Cost_t(self, t, x, d, z, delta, Sigma):
        """
        input:
            t - 時刻
            x - 時刻tにおける主体の残存注文量X_t
            d - 時刻tにおける価格偏差D_t
            z - 時刻tにおけるファンダメンタル価格S_t
            delta - 時刻tにおけるintensityの差kappa_t^+ - kappa_t^-
            Sigma - 時刻tにおけるintensityの和kappa_t^+ + kappa_t^-
        """
        first_block = -self.q*(z+d)*x+((1-self.epsilon)/(2+self.rho*(self.T-t))+self.epsilon/2)*x**2
        second_block = -1/(1-self.epsilon)*(self.rho*(self.T-t)/2)/(2+self.rho*(self.T-t)) \
                    *(self.q*d-self.calG_ita(self.T-t)*delta*self.m_1/self.rho)*x
        third_block = -1/(1-self.epsilon)*(self.rho*(self.T-t)/2)/(2+self.rho*(self.T-t)) \
                    *(self.q*d-self.calG_ita(self.T-t)*delta*self.m_1/self.rho)**2
        forth_block = self.c_hat_ita(self.T-t)*(delta*self.m_1/self.rho)**2 \
                    +self.e(self.T-t)*Sigma+self.g(self.T-t)
        return (first_block + second_block + third_block + forth_block)/self.q

if __name__ == '__main__':
    data = {'q':100,'T':1,'nu':0.3,'epsilon':0.3,'rho':25,'iota_c':2,'kappa_inf':12, \
            'ita':0,'m_1':50,'m_2':100,'alpha_tilda':5,'alpha_2':5,'beta':20}
    cost = Cost()
    cost.set_up(data)
    print(cost.Cost_t(0.5, -200, 10, 100, 10, 10))