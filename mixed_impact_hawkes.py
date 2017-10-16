#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:10:03 2017

@author: okadatomoki
"""

import numpy as np
from scipy.integrate import quad

class MIH():

    def set_up(self, data):
        """
        パラメータのセットアップ
        """
        self.q = data['q']
        self.T = data['T']
        self.n = data['n']
        self.x_0 = data['x_0']
        self.delta_0 = data['delta_0']
        self.D_0 = data['D_0']
        self.nu = data['nu']
        self.epsilon = data['epsilon']
        self.rho = data['rho']
        self.iota_s = data['iota_s']
        self.iota_c = data['iota_c']
        self.kappa_inf = data['kappa_inf']
        self.ita = data['ita']
        self.m_1 = data['m_1']
        self.m_2 = data['m_2']
        self.alpha_tilda = data['alpha_tilda']
        self.alpha_2 = data['alpha_2']
        self.beta = data['beta']
        self.Delta_N = [0,3,5,3,-1,-0,1,3,7,-1,3]
        self.tau = [0,1,2,3,4,5,6,7,8,9,10]
        self.kappa_plus = [2,4,2,4,5,6,4,8,8,2,1]
        self.kappa_minus = [2,4,4,6,2,4,7,8,2,4,2]
    
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

#==============================================================================

    def phi_ita(self, t):
        return 1/(2*(2+self.rho*(self.T-t)))*(1+np.exp(-self.ita*(self.T-t)) \
                    +self.nu*self.rho*(self.T-t)*self.zeta(self.ita*(self.T-t)) \
                    +self.beta/self.rho(2+self.rho*(self.T-t)*(1+self.zeta(self.ita*(self.T-t)) \
                    +self.nu*self.rho*(self.T-t)*self.omega(self.ita*(self.T-t)))))
    
    def Phi_0(self, s,t):
        first_term = (self.beta/self.rho+self.nu/2*(1/2-self.beta/self.rho))*(np.exp(-self.beta*s)-np.exp(-self.beta*t))/self.beta
        second_term = (1-self.nu)*(1-self.beta/self.rho)*np.exp(-self.beta*self.T) \
                    /self.rho*(self.L(self.rho,self.beta,self.T-s)-self.L(self.rho,self.beta,self.T-t))
        third_term = self.nu/4*((self.T-t)*np.exp(-self.beta*s))-(self.T-t)*np.exp(-self.beta*t)
        return first_term + second_term + third_term
        
    def Phi_ita(self, s,t):
        first_term = 1/2*(1/self.rho+self.nu/self.ita)*(np.exp(-self.beta*s)-np.exp(-self.beta*t))
        second_term = np.exp(-self.beta*self.T)/(2*self.rho)*(1+self.nu*(self.rho-2*self.beta) \
                    /self.ita+self.beta/self.ita*(1-self.nu*self.rho/self.ita)) \
                    *(self.L(self.rho,self.beta,self.T-s)-self.L(self.rho,self.beta,self.T-t))
        third_term = np.exp(-self.beta*self.T)/(2*self.rho)*(1-self.nu*self.rho/self.ita-self.beta/self.ita \
                    *(1-self.nu*self.rho/self.ita))*(self.L(self.rho,self.alpha,self.T-s)-self.L(self.rho,self.alpha,self.T-t))
        return first_term + second_term + third_term
        
    def delta_t(self, t):
        return self.kappa_plus[t] - self.kappa_minus[t]
        
    def Theta_i(self, i):
        Delta_I = (self.iota_s - self.iota_c)*self.Delta_N
        _sum = 0
        for k in range(i):
            _sum += np.exp(self.beta*self.tau[k])*Delta_I[i]
        return _sum
        
    def Delta_X_0_OW(self):
        return -self.x_0/(2*self.rho*self.T)
    
    def Delta_X_T_OW(self):
        return -self.x_0/(2*self.rho*self.T)
        
    def Delta_X_t_OW(self):
        dt = self.T/self.n
        return -self.rho*self.x_0*dt/(2+self.rho*self.T)
           
    def Delta_X_trend_0(self):
        return (self.delta_0*self.m_1/2*self.rho*(2+self.rho*self.T*(1+self.zeta(self.ita*self.T) \
                +self.nu*self.rho*self.T*self.omega(self.ita*self.T)))-(1+self.rho*self.T)*self.q*self.D_0) \
                /((2+self.rho*self.T)*(1-self.epsilon))
        
    def Delta_X_trend_T(self):
        first_term = self.delta_0*self.m_1/(2*self.rho)*((2+self.rho*self.T*(1+self.zeta(self.ita*self.T) \
                    +self.nu*self.rho*self.T*self.omega(self.ita*self.T)))/(2*self.rho*self.T)-2*self.rho*self.Phi_ita(0,self.T) \
                    -2*np.exp(-self.beta*self.T))
        second_term = self.q*self.D_0/(2+self.rho*self.T)
        return (first_term + second_term)/(1 - self.epsilon)
    
    def Delta_X_trend_t(self, t):
        dt = self.T/self.n
        first_term = self.delta_0*self.m_1/(2*self.rho)*((2+self.rho*self.T*(1+self.zeta(self.ita*self.T) \
                    +self.nu*self.rho*self.T*self.omega(self.ita*self.T)))/(2*self.rho*self.T)-2*self.rho*self.Phi_ita(0,t) \
                    -2*self.phi_ita(t)*np.exp(-self.beta*t))*self.rho*dt 
        second_term = self.q*self.D_0/(2+self.rho*self.T)*self.rho*dt
        return (first_term + second_term)/(1 - self.epsilon)
        
    def Delta_X_dyn_0(self):
        return 0
        
    def Delta_X_dyn_T(self):
        Delta_I = (self.iota_s - self.iota_c)*self.Delta_N
        first_term = -self.m_1(self.Theta_i(self.T/self.n)*self.Phi_ita(self.T/self.n,self.T) \
                    +sum(self.Theta_i(i)*self.Phi_ita(self.tau[i],self.tau[i+1]) for i in range(0, self.T/self.n-1)))
        second_term = sum((1-self.nu)*self.Delta_N[i]/(2+self.rho*(self.T-self.tau[i])) for i in range(len(self.tau)))
        third_term = self.m_1/(2*self.rho)*sum((2+self.rho*(self.T-self.tau[i]) \
                    *(1+self.zeta(self.ita*(self.T-self.tau[i])+self.nu*self.rho*(self.T-self.tau[i]) \
                    *self.omega(self.ita*(self.T-self.tau[i])))))/(2+self.rho*(self.T-self.tau[i]))*Delta_I[i] for i in range(len(self.tau)))
        forth_term = -self.m_1/self.rho*self.Theta_i(self.T/self.n)*np.exp(-self.beta*self.T)
        return first_term + second_term + third_term + forth_term
        
    def Delta_X_dyn_t(self, t):
        dt = self.T/self.n
        Delta_I = (self.iota_s - self.iota_c)*self.Delta_N
        first_term = -self.m_1*self.phi_ita(t)*self.Theta_i(t)*np.exp(-self.beta*t)*dt
        second_term = (sum((1-self.nu)*self.Delta_N[i]/(2+self.rho*(self.T-self.tau[i])) for i in range(t)))*self.rho*dt
        third_term = (sum((2+self.rho*(self.T-self.tau[i]) \
                    *(1+self.zeta(self.ita*(self.T-self.tau[i])+self.nu*self.rho*(self.T-self.tau[i]) \
                    *self.omega(self.ita*(self.T-self.tau[i])))))/(2+self.rho*(self.T-self.tau[i]))*Delta_I[i] for i in range(len(t))))*self.m_1/2*dt
        forth_term = -(self.Theta_i(t)*self.Phi_ita(self.tau[t], t) \
                    +sum(self.Theta_i(i)*self.Phi_ita(self.tau[i],self.tau[i+1]) for i in range(0, t/self.n-1)))*self.rho*self.m_1*dt
        fifth_term = (1+self.rho*(self.T-t))/(2+self.rho*(self.T-t))*(self.m_1 \
                    /self.rho*self.Delta_I[t]-(1-self.nu)*self.Delta_N[t])
        sixth_term = self.m_1/(2*self.rho)*(self.nu*self.rho-self.ita)*(self.rho*(self.T-t)**2 \
                    *self.omega(self.ita*(self.T-t)))/(2+self.rho*(self.T-t))*Delta_I[t]
        return (first_term + second_term + third_term + forth_term + fifth_term + sixth_term)/(1 - self.epsilon)

if __name__ == '__main__':
    data = {'q':100,'T':1,'n':10,'x_0':-500,'nu':0.3,'epsilon':0.3,'rho':25,'iota_c':2,'kappa_inf':12, \
            'ita':0,'m_1':50,'m_2':100,'alpha_tilda':5,'alpha_2':5,'beta':20}
    cost = MIH()
    cost.set_up(data)
    print(cost.Cost_t(0.5, -200, 10, 100, 10, 10))