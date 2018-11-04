    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm

'''
Asymptotic approximation for 0<beta<=1 by Hagan
'''
def bsm_vol(strike, forward, texp, sigma, alpha=0, rho=0, beta=1):
    if(texp<=0.0):
        return( 0.0 )

    powFwdStrk = (forward*strike)**((1-beta)/2)
    logFwdStrk = np.log(forward/strike)
    logFwdStrk2 = logFwdStrk**2
    rho2 = rho*rho

    pre1 = powFwdStrk*( 1 + (1-beta)**2/24 * logFwdStrk2*(1 + (1-beta)**2/80 * logFwdStrk2) )
  
    pre2alp0 = (2-3*rho2)*alpha**2/24
    pre2alp1 = alpha*rho*beta/4/powFwdStrk
    pre2alp2 = (1-beta)**2/24/powFwdStrk**2

    pre2 = 1 + texp*( pre2alp0 + sigma*(pre2alp1 + pre2alp2*sigma) )

    zz = powFwdStrk*logFwdStrk*alpha/np.fmax(sigma, 1e-32)  # need to make sure sig > 0
    if isinstance(zz, float):
        zz = np.array([zz])
    yy = np.sqrt(1 + zz*(zz-2*rho))

    xx_zz = np.zeros(zz.size)

    ind = np.where(abs(zz) < 1e-5)
    xx_zz[ind] = 1 + (rho/2)*zz[ind] + (1/2*rho2-1/6)*zz[ind]**2 + 1/8*(5*rho2-3)*rho*zz[ind]**3
    ind = np.where(zz >= 1e-5)
    xx_zz[ind] = np.log( (yy[ind] + (zz[ind]-rho))/(1-rho) ) / zz[ind]
    ind = np.where(zz <= -1e-5)
    xx_zz[ind] = np.log( (1+rho)/(yy[ind] - (zz[ind]-rho)) ) / zz[ind]

    bsmvol = sigma*pre2/(pre1*xx_zz) # bsm vol
    return(bsmvol[0] if bsmvol.size==1 else bsmvol)

'''
Asymptotic approximation for beta=0 by Hagan
'''
def norm_vol(strike, forward, texp, sigma, alpha=0, rho=0):
    # forward, spot, sigma may be either scalar or np.array. 
    # texp, alpha, rho, beta should be scholar values

    if(texp<=0.0):
        return( 0.0 )
    
    zeta = (forward - strike)*alpha/np.fmax(sigma, 1e-32)
    # explicitly make np.array even if args are all scalar or list
    if isinstance(zeta, float):
        zeta = np.array([zeta])
        
    yy = np.sqrt(1 + zeta*(zeta - 2*rho))
    chi_zeta = np.zeros(zeta.size)
    
    rho2 = rho*rho
    ind = np.where(abs(zeta) < 1e-5)
    chi_zeta[ind] = 1 + 0.5*rho*zeta[ind] + (0.5*rho2 - 1/6)*zeta[ind]**2 + 1/8*(5*rho2-3)*rho*zeta[ind]**3

    ind = np.where(zeta >= 1e-5)
    chi_zeta[ind] = np.log( (yy[ind] + (zeta[ind] - rho))/(1-rho) ) / zeta[ind]

    ind = np.where(zeta <= -1e-5)
    chi_zeta[ind] = np.log( (1+rho)/(yy[ind] - (zeta[ind] - rho)) ) / zeta[ind]

    nvol = sigma * (1 + (2-3*rho2)/24*alpha**2*texp) / chi_zeta
 
    return(nvol[0] if nvol.size==1 else nvol)

'''
Hagan model class for 0<beta<=1
'''
class ModelHagan:
    alpha, beta, rho = 0.0, 1.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.beta = beta
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return bsm_vol(strike, forward, texp, sigma, alpha=self.alpha, beta=self.beta, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        bsm_vol = self.bsm_vol(strike, spot, texp, sigma)
        return self.bsm_model.price(strike, spot, texp, bsm_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        return sigma
    
    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):

        '''  
        Given option prices or bsm vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp = self.texp if(texp is None) else texp
        if isinstance(strike3, (int, float)):
            strike3 = [strike3]
        
        if isinstance(price_or_vol3, (int, float)):
            price_or_vol3 = [price_or_vol3]
        
        if(is_vol):
            vol = price_or_vol3
        else:
            vol = [self.bsm_model.impvol(price_or_vol3[i], strike3[i], spot, texp, cp_sign=cp_sign) for i in range(3)]

        def FOC(x):
            # set constraints to the parameters
            sigma = np.sqrt(x[0]**2)
            alpha = np.sqrt(x[1]**2)
            vol = [self.bsm_model.impvol(price_or_vol3[i], strike3[i], spot, texp, cp_sign) for i in range(len(strike3))]
        
        def func(x):  # x is a vector of 3 scallers for sigma, alpha, rho respectively.
            sigma = np.abs(x[0])
            alpha = np.abs(x[1])
            rho = 2*x[2]/(1+x[2]**2)
            return [ bsm_vol(strike3[i], spot, texp, sigma, alpha, rho) - vol[i] for i in range(len(strike3)) ]
        
        sol_root = sopt.root(func, np.zeros(len(strike3)) )
        solution = sol_root.x

        sigma = np.abs(solution[0])
        alpha = np.abs(solution[1])
        rho = 2 * solution[2] / (1 + solution[2]**2)
        
        return sigma, alpha, rho # sigma, alpha, rho

'''
Hagan model class for beta=0
'''
class ModelNormalHagan:
    alpha, beta, rho = 0.0, 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.beta = 0.0 # not used but put it here
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return norm_vol(strike, forward, texp, sigma, alpha=self.alpha, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        n_vol = self.norm_vol(strike, spot, texp, sigma)
        return self.normal_model.price(strike, spot, texp, n_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            norm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 50)
        if(setval):
            self.sigma = sigma
        return sigma

    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or normal vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp = self.texp if(texp is None) else texp
        
        if(is_vol):
            vol = price_or_vol3
        else:
            vol = [self.normal_model.impvol(price_or_vol3[i], strike3[i], spot, texp, cp_sign=cp_sign) for i in range(3)]


        def FOC(x):
            # set constraints to the parameters
            sigma = np.sqrt(x[0]**2)
            alpha = np.sqrt(x[1]**2)
            rho = 2*x[2]/(1+x[2]**2)
            norm_vol1 = norm_vol(strike3[0], spot, texp, sigma, alpha=alpha, rho=rho) - vol[0]
            norm_vol2 = norm_vol(strike3[1], spot, texp, sigma, alpha=alpha, rho=rho) - vol[1]
            norm_vol3 = norm_vol(strike3[2], spot, texp, sigma, alpha=alpha, rho=rho) - vol[2]
            return [norm_vol1,norm_vol2,norm_vol3]
            
        sol_root = sopt.root(FOC,np.array([0,0,0]))
        solution_x = sol_root.x

        sigma = np.sqrt(solution_x[0]**2)
        alpha = np.sqrt(solution_x[1]**2)
        rho = 2*solution_x[2]/(1+solution_x[2]**2)

        return sigma, alpha, rho # sigma, alpha, rho

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    time_steps = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0,time_steps=120,n_samples=1000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''

        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp=texp, sigma=sigma)
        vol = self.bsm_model.impvol(price, strike, spot, texp)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        
        return sigma
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        # np.random.seed(12345)
        if isinstance(strike, int or float):
            strike = np.array([strike]) 

        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        div_fac = np.exp(texp*self.divr)
        disc_fac = np.exp(texp*self.intr)
        forward = spot / disc_fac * div_fac

        forward = forward * np.ones(strike.size)
        
        delta_t = texp / self.time_steps
        znorm_m = np.random.normal(size=(strike.size, self.time_steps, 2, self.n_samples))

        Z1 = znorm_m[:, :, 0]
        Z2 = self.rho * znorm_m[:, :, 0] + np.sqrt(1 - self.rho ** 2) * znorm_m[:, :, 1]

        temp_delta = np.exp(self.alpha * np.sqrt(delta_t) * Z2 - 0.5 * self.alpha ** 2 * delta_t)

        delta_k = sigma * np.cumprod(temp_delta, axis=1)

        S_T = forward[:, np.newaxis] * np.cumprod(np.exp(delta_k * np.sqrt(delta_t) * Z1 - 0.5 * delta_k**2 * delta_t), axis=1)[:,-1]

        return disc_fac*np.mean(np.fmax(cp_sign*(S_T - strike[:, np.newaxis]), 0), axis=1)

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0,time_steps=120,n_samples=1000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp=texp, sigma=sigma)
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            normal_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 100)
        if(setval):
            self.sigma = sigma
        
        return sigma

        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''

        # np.random.seed(12345)
        if isinstance(strike, int or float):
            strike = np.array([strike]) 

        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac

        forward = forward * np.ones(strike.size)
        
        delta_t = texp / self.time_steps
        znorm_m = np.random.normal(size=(strike.size, self.time_steps, 2, self.n_samples))

        Z1 = znorm_m[:, :, 0]
        Z2 = self.rho * znorm_m[:, :, 0] + np.sqrt(1 - self.rho ** 2) * znorm_m[:, :, 1]

        temp_delta = np.exp(self.alpha * np.sqrt(delta_t) * Z2 - 0.5 * self.alpha ** 2 * delta_t)

        delta_k = sigma * np.cumprod(temp_delta, axis=1)

        S_T = forward[:, np.newaxis] + np.cumsum(delta_k * np.sqrt(delta_t) * Z1, axis=1)[:,-1]

        return disc_fac*np.mean(np.fmax(cp_sign*(S_T - strike[:, np.newaxis]), 0), axis=1)

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0,time_steps=120,n_samples=1000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''

        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp=texp, sigma=sigma)
        vol = self.bsm_model.impvol(price, strike, spot, texp)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        
        return sigma

    def generate_S0_Sigma(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        texp = self.texp if texp is None else texp
        sigma = self.sigma if sigma is None else sigma
        np.random.seed(12345)
        z = np.random.normal(size = (strike.size, self.time_steps,self.n_samples))
        delta_t = texp/self.time_steps

        sigma_path = np.exp(-0.5* self.alpha**2 * delta_t + self.alpha * np.sqrt(delta_t) * z)

        delta_sigma = sigma * np.cumprod(sigma_path, axis=1)
        sigma_T = delta_sigma[:,-1]
        new_S0 = spot+self.rho/self.alpha*(sigma_T-sigma)
        I_T = np.sum(delta_sigma*delta_t,axis=1)
        new_sigma = np.sqrt((1-(self.rho**2)*I_T/texp))
        return new_S0,new_sigma

    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        texp = self.texp if texp is None else texp
        sigma = self.sigma if sigma is None else sigma
        
        new_S0,new_sigma = self.generate_S0_Sigma(strike, spot, texp=texp, sigma=sigma, cp_sign=cp_sign)
        mc_price = [np.mean(self.bsm_model.price(strike[i], new_S0[i], texp, new_sigma[i], cp_sign)) for i in range(strike.size)]
        return mc_price
        np.random.seed(12345)
        n_step = self.time_steps
        n_sample = self.n_samples
        
        if isinstance(strike, int or float):
            strike = [strike] 
        
        price_array = np.zeros((len(strike), n_sample))
        z = np.random.normal(size = (n_sample, n_step))
        
        for j in range(n_sample):  # simulate for n_simu times
            
            sigma_path = [np.exp(-0.5* self.alpha**2 * texp/n_step + self.alpha * np.sqrt(texp/n_step) * z[j, t]) for t in range(n_step)]
            sigma_path.insert(0, sigma)
            sigma_path = np.cumprod(sigma_path)
            
            I = np.sum(sigma_path ** 2 * texp/n_step)
            new_spot = spot*np.exp(self.rho * (sigma_path[-1] -  sigma) / self.alpha - self.rho ** 2 / 2 * I)
            new_sigma = np.sqrt((1-self.rho ** 2) * I / texp)
            price_array[:, j] = self.bsm_model.price(strike, new_spot, texp, new_sigma, cp_sign)  # replace the jth column
        
        return np.mean(price_array, 1)

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0,time_steps=120,n_samples=1000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp=texp, sigma=sigma)
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            normal_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 100)
        if(setval):
            self.sigma = sigma
        
        return sigma
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        texp = self.texp if texp is None else texp
        sigma = self.sigma if sigma is None else sigma
        # np.random.seed(12345)
        n_step = self.time_steps
        n_sample = self.n_samples
        
        if isinstance(strike, int or float):
            strike = [strike] 
        
        price_array = np.zeros((len(strike), n_sample))
        z = np.random.normal(size = (n_sample, n_step))
        
        for j in range(n_sample):
            
            sigma_path = [np.exp(-0.5* self.alpha**2 * texp/n_step + self.alpha * np.sqrt(texp/n_step) * z[j, t]) for t in range(n_step)]
            sigma_path.insert(0, sigma)
            sigma_path = np.cumprod(sigma_path)
            
            I = np.sum(sigma_path ** 2 * texp/n_step)
            
            new_spot = spot + self.rho * (sigma_path[-1] -  sigma) / self.alpha
            new_sigma = np.sqrt((1-self.rho ** 2) * I / texp)
            price_array[:, j] = self.normal_model.price(strike, new_spot, texp, new_sigma, cp_sign)  # replace the jth column
        
        return np.mean(price_array, 1)
