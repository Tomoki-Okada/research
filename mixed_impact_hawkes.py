#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:10:03 2017

@author: okadatomoki
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad

class MIH():

    def parameter_set_up(self, para):
        """
        パラメータのセットアップ
        """
        self.q = para['q']
        self.T = 1
        self.n = 10
        self.tau = [self.T/self.n * i for i in range(self.n + 1)]
        self.dt = self.T/(self.n-1)
        self.inter_periods = self.n - 2
        self.x_0 = para['x_0']
        self.delta_0 = para['delta_0']
        self.D_0 = para['D_0']
        self.nu = para['nu']
        self.epsilon = para['epsilon']
        self.rho = para['rho']
        self.iota_s = para['iota_s']
        self.iota_c = para['iota_c']
        self.kappa_inf = para['kappa_inf']
        self.ita = 0
        self.m_1 = para['m_1']
        self.m_2 = para['m_2']
        self.A = np.array([[self.iota_s, self.iota_c],[self.iota_c, self.iota_s]])
        self.alpha = self.iota_s - self.iota_c
        self.alpha_tilda = para['alpha_tilda']
        self.alpha_2 = para['alpha_2']
        self.beta = para['beta']

    def csv_load(self, path):
        self.event_data = pd.read_csv(path, names = ['time', 'mark'])
        
    def data_setup(self):
        def Delta_N():
            """
            input:
                data(第1列がタイムスタンプで第2列が注文属性のマーク)
            output:
                Delta_Nのリスト
            """
            _Delta_N = []
            for i in range(1, 11):
                block = self.event_data[0.1 * (i-1) <= self.event_data['time']][self.event_data['time'] < 0.1*i]
                N_plus = len(block[block['mark'] == 1])
                N_minus = len(block[block['mark'] == 2])
                N = N_plus - N_minus
                _Delta_N.append(N)
            _Delta_N = np.append(0,_Delta_N)
            return _Delta_N
        
        def time_list():
            first_list = list(self.event_data[self.event_data['mark'] == 1]['time'])
            second_list = list(self.event_data[self.event_data['mark'] == 2]['time'])
            return [first_list, second_list]
        
        def Intensity(t, s):
            """
            input:
                lambda_0:2次元ベース強度
                alpha:2×2次元ndarray行列
                beta:2×2次ndarray元行列
                t:2次元の生起時間多重リスト
                s:時刻
            output:
                時刻sでのintensity
            """
            intensity = np.zeros(2)
            lambda_0 = [self.kappa_inf, self.kappa_inf]
            for m in range(2):
                intensity[m] += lambda_0[m]
                for n in range(2):
                    for i in range(len(t[n])):
                        if t[n][i] < s: 
                            intensity[m] += self.A[n,m] * np.exp(-self.beta*(s-t[n][i]))
            return intensity
        
        intensity = Intensity(time_list(), 0)
        for tau_i in self.tau[1:]:
            intensity = np.vstack((intensity, Intensity(time_list(), tau_i)))
        
        self.Delta_N = Delta_N()
        self.Delta_I = (self.iota_s - self.iota_c) * self.Delta_N
        self.kappa_plus = intensity[:,0]
        self.kappa_minus = intensity[:,1]
        self.delta_t = self.kappa_plus - self.kappa_minus
                
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

#================================================================================================================================

    def phi_ita(self, t):
        return 1/(2*(2+self.rho*(self.T-t)))*(1+np.exp(-self.ita*(self.T-t)) \
                    +self.nu*self.rho*(self.T-t)*self.zeta(self.ita*(self.T-t)) \
                    +self.beta/self.rho*(2+self.rho*(self.T-t)*(1+self.zeta(self.ita*(self.T-t)) \
                    +self.nu*self.rho*(self.T-t)*self.omega(self.ita*(self.T-t)))))
    
    def Phi_ita(self, s, t):
        if self.ita == 0:   
            first_term = (self.beta/self.rho+self.nu/2*(1/2-self.beta/self.rho))*(np.exp(-self.beta*s)-np.exp(-self.beta*t))/self.beta
            second_term = (1-self.nu)*(1-self.beta/self.rho)*np.exp(-self.beta*self.T) \
                        /self.rho*(self.L(self.rho,self.beta,self.T-s)-self.L(self.rho,self.beta,self.T-t))
            third_term = self.nu/4*((self.T-t)*np.exp(-self.beta*s))-(self.T-t)*np.exp(-self.beta*t)
            return first_term + second_term + third_term
        else:
            first_term = 1/2*(1/self.rho+self.nu/self.ita)*(np.exp(-self.beta*s)-np.exp(-self.beta*t))
            second_term = np.exp(-self.beta*self.T)/(2*self.rho)*(1+self.nu*(self.rho-2*self.beta) \
                        /self.ita+self.beta/self.ita*(1-self.nu*self.rho/self.ita)) \
                        *(self.L(self.rho,self.beta,self.T-s)-self.L(self.rho,self.beta,self.T-t))
            third_term = np.exp(-self.beta*self.T)/(2*self.rho)*(1-self.nu*self.rho/self.ita-self.beta/self.ita \
                        *(1-self.nu*self.rho/self.ita))*(self.L(self.rho,self.alpha,self.T-s)-self.L(self.rho,self.alpha,self.T-t))
            return first_term + second_term + third_term            
        
    def Theta_i(self, i):
        if i == 0:
            return 0
        else:
            _Theta_i = sum([np.exp(self.beta * self.tau[l]) * self.Delta_I[l] for l in range(1,i+1)])
            return _Theta_i
        
    def Delta_X_OW(self): 
        """
        OWモデルの最適執行戦略
        """  
        def Delta_X_OW_0():
            return -self.x_0/(2+self.rho*self.T)
        
        def Delta_X_OW_T():
            return -self.x_0/(2+self.rho*self.T)
            
        def Delta_X_OW_t():
            return -self.rho*self.x_0*self.dt/(2+self.rho*self.T)
        
        _Delta_X_OW = Delta_X_OW_t() * np.ones(self.inter_periods)
        _Delta_X_OW = np.append(Delta_X_OW_0(), _Delta_X_OW)
        _Delta_X_OW = np.append(_Delta_X_OW, Delta_X_OW_T())
        return _Delta_X_OW        
        
    def Delta_X_trend(self):
        """
        trend部分の最適執行戦略
        """
        def Delta_X_trend_0():
            return (self.delta_0*self.m_1/2*self.rho*(2+self.rho*self.T*(1+self.zeta(self.ita*self.T) \
                    +self.nu*self.rho*self.T*self.omega(self.ita*self.T)))-(1+self.rho*self.T)*self.q*self.D_0) \
                    /((2+self.rho*self.T)*(1-self.epsilon))
            
        def Delta_X_trend_T():
            first_term = self.delta_0*self.m_1/(2*self.rho)*((2+self.rho*self.T*(1+self.zeta(self.ita*self.T) \
                        +self.nu*self.rho*self.T*self.omega(self.ita*self.T)))/(2*self.rho*self.T)-2*self.rho*self.Phi_ita(0,self.T) \
                        -2*np.exp(-self.beta*self.T))
            second_term = self.q*self.D_0/(2+self.rho*self.T)
            return (first_term + second_term)/(1 - self.epsilon)
        
        def Delta_X_trend_t(t):
            first_term = self.delta_0*self.m_1/(2*self.rho)*((2+self.rho*self.T*(1+self.zeta(self.ita*self.T) \
                        +self.nu*self.rho*self.T*self.omega(self.ita*self.T)))/(2*self.rho*self.T)-2*self.rho*self.Phi_ita(0,t) \
                        -2*self.phi_ita(t)*np.exp(-self.beta*t))*self.rho*self.dt 
            second_term = self.q*self.D_0/(2+self.rho*self.T)*self.rho*self.dt
            return (first_term + second_term)/(1 - self.epsilon)
            
        _Delta_X_trend = np.array([Delta_X_trend_t(0.1 * i) for i in range(1,10)])
        _Delta_X_trend = np.append(Delta_X_trend_0(), _Delta_X_trend)
        _Delta_X_trend = np.append(_Delta_X_trend, Delta_X_trend_T())
        return _Delta_X_trend
        
    def Delta_X_dyn(self):
        """
        dynamic部分の最適執行戦略
        """
        def Delta_X_dyn_0():
            return 0
            
        def Delta_X_dyn_T():
            first_term = -self.m_1*(self.Theta_i(self.n)*self.Phi_ita(self.T,self.T) \
                        +sum([self.Theta_i(i)*self.Phi_ita(self.tau[i],self.tau[i+1]) for i in range(1, self.n)]))
            second_term = sum([(1-self.nu)*self.Delta_N[i]/(2+self.rho*(self.T-self.tau[i])) for i in range(1,len(self.tau))])
            third_term = self.m_1/(2*self.rho)*sum([(2+self.rho*(self.T-self.tau[i]) \
                        *(1+self.zeta(self.ita*(self.T-self.tau[i])+self.nu*self.rho*(self.T-self.tau[i]) \
                        *self.omega(self.ita*(self.T-self.tau[i])))))/(2+self.rho*(self.T-self.tau[i]))*self.Delta_I[i] for i in range(len(self.tau))])
            forth_term = -self.m_1/self.rho*self.Theta_i(self.n)*np.exp(-self.beta*self.T)
            return first_term + second_term + third_term + forth_term
            
        def Delta_X_dyn_t(t):
            k = int(t/0.1)
            first_term = -self.m_1*self.phi_ita(t)*self.Theta_i(k)*np.exp(-self.beta*t)*self.dt
            second_term = sum([(1-self.nu)*self.Delta_N[i]/(2+self.rho*(self.T-self.tau[i])) for i in range(1,k+1)])*self.rho*self.dt
            third_term = (sum([(2+self.rho*(self.T-self.tau[i]) \
                        *(1+self.zeta(self.ita*(self.T-self.tau[i])+self.nu*self.rho*(self.T-self.tau[i]) \
                        *self.omega(self.ita*(self.T-self.tau[i])))))/(2+self.rho*(self.T-self.tau[i]))*self.Delta_I[i] for i in range(1,k+1)]))*self.m_1/2*self.dt
            forth_term = -(self.Theta_i(k)*self.Phi_ita(self.tau[k], t) \
                        +sum([self.Theta_i(i)*self.Phi_ita(self.tau[i],self.tau[i+1]) for i in range(1, k)]))*self.rho*self.m_1*self.dt
            fifth_term = (1+self.rho*(self.T-t))/(2+self.rho*(self.T-t))*(self.m_1 \
                        /self.rho*self.Delta_I[k]-(1-self.nu)*self.Delta_N[k])
            sixth_term = self.m_1/(2*self.rho)*(self.nu*self.rho-self.ita)*(self.rho*(self.T-t)**2 \
                        *self.omega(self.ita*(self.T-t)))/(2+self.rho*(self.T-t))*self.Delta_I[k]
            return (first_term + second_term + third_term + forth_term + fifth_term + sixth_term)/(1 - self.epsilon)
            
        _Delta_X_dyn = np.array([Delta_X_dyn_t(0.1 * i) for i in range(1,10)])
        _Delta_X_dyn = np.append(Delta_X_dyn_0(), _Delta_X_dyn)
        _Delta_X_dyn = np.append(_Delta_X_dyn, Delta_X_dyn_T())
        return _Delta_X_dyn
        
    def Delta_X_dyn_tau(self, t):
        k = int(t/0.1)
        first_term = (1+self.rho*(self.T-self.tau[k]))/(2+self.rho*(self.T-self.tau[k]))*(self.m_1/self.rho*self.Delta_I[k]-(1-self.nu)*self.Delta_N[k])
        second_term = self.m_1/(2*self.rho)*(self.nu*self.rho-self.ita)*self.rho*(self.T-self.tau[k])**2*self.omega(self.ita*(self.T-self.tau[k]))/(2+self.rho*(self.T-self.tau[k]))*self.Delta_I[k]
        return (first_term + second_term)/(1-self.epsilon)   

    def special_Delta_X_dyn_tau(self, t):
        k = int(t/0.1)
        term = (-1)*(1+self.rho*(self.T-self.tau[k]))/(2+self.rho*(self.T-self.tau[k]))*(1-self.nu)*self.Delta_N(k)
        return term/(1-self.epsilon)

if __name__ == '__main__':
    data = {'q':100,'T':1,'n':10,'x_0':-500,'nu':0.3,'epsilon':0.3,'rho':25,'iota_c':2,'kappa_inf':12, \
            'ita':0,'m_1':50,'m_2':100,'alpha_tilda':5,'alpha_2':5,'beta':20}
    cost = MIH()
    cost.set_up(data)
    print(cost.Cost_t(0.5, -200, 10, 100, 10, 10))
    
    mih = MIH()
    mih.n = 10
    mih.T = 1
    mih.inter_periods = 9
    mih.rho = 25
    mih.beta = 20
    mih.x_0 = -500
    mih.ita = 0
    mih.nu = 0.3
    mih.epsilon = 0.3
    mih.delta_0 = 0
    mih.m_1 = 50
    mih.dt = 1/9
    mih.q = 100
    mih.D_0 = 0.1
    mih.tau = [mih.T/mih.n * i for i in range(mih.n + 1)]
    mih.kappa_inf = 12
    mih.iota_s = 16
    mih.iota_c = 2
    mih.A = np.array([[mih.iota_s, mih.iota_c],[mih.iota_c, mih.iota_s]])
    mih.csv_load('test.csv')
    mih.data_setup()
    mih.Delta_X_OW()
    