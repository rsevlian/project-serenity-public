import numpy as np 
from scipy import stats 
from scipy.stats import norm 

import multiprocessing 
from multiprocessing import Pool 
  
import mining_constants
# CONTRACT_CALL_OPTION = "C" 
# CONTRACT_PUT_OPTION = "P" 
  

def bsm_components(sigma: float, S: float, K: float, r: float, T: float) -> float: 
    """
        Computes d1, d2 coefficients for BSM model.
    """  

    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T) 
  
    return d1, d2 


def bsm_call_price(sigma: float, S: float, K: float, r: float, T: float) -> float: 
    """
        BSM Call Price Calculator:
        
        input:
            sigma - annualized volatility
            s     - asset spot price
            K     - strike price
            r     - risk free interest rate
            t     - time to maturity of option 

        output:
            call price in USD
    """
    if(type(T) is int and T == 0): 
        return max(S - K, 0) 
    else:
        # extract args. 
        d1, d2 = bsm_components(sigma, S, K, r, T)
        call_price = np.exp(-r * T) * (S * np.exp(r * T) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)) 
  
    return call_price 
  
  
def bsm_put_price(sigma: float, S: float, K: float, r: float, T: float) -> float:
  
    """    
        BSM Price Price Calculator:
        input:
            sigma - annualized volatility
            s     - asset spot price
            K     - strike price
            r     - risk free interest rate
            t     - time to maturity of option 
        
        output:
            put price in USD
    """
    if(type(T) is int and T == 0): 
        return max(K - S, 0) 
    
    else:
        d1, d2 = bsm_components(sigma, S, K, r, T)
        price = np.exp(-r * T) * (K * stats.norm.cdf(-d2) - S * np.exp(r * T) * stats.norm.cdf(-d1)) 
  
    return price 
  
# Generic Option Pricing Functions.
def bsm_price(sigma: float, S: float, K: float, r: float, T: float, contract_type: str) -> float:
    """
        BSM Price Price Calculator:
        
        input:
            sigma - annualized volatility [1.0 = 100%]
            s     - asset spot price
            K     - strike price
            r     - risk free interest rate
            t     - time to maturity of option 

        output:
            BSM price in USD.
    """
    if (contract_type == mining_constants.CONTRACT_CALL_OPTION): 
        return bsm_call_price(sigma, S, K, r, T) 
    
    elif (contract_type == mining_constants.CONTRACT_PUT_OPTION):
        return bsm_put_price(sigma, S, K, r, T) 
    
    else:
        raise ValueError(f"Invalid contract type {contract_type}") 
   

def bsm_delta(sigma: float, S: float, K: float, T: float, r: float, contract_type: str) -> float: 
    """
        Compute Option Delta:

        input:
            sigma - annualized volatility [1.0 = 100%].
            S     - spot price of underlying [USD].
            K     - option strike price [USD].
        output:
            option delta 
    """
    
    # Options at expiry should have 0, +- 1 delta 
    if(type(T) is int and T == 0): 
        if(contract_type == CONTRACT_CALL_OPTION): 
            return 1 * (S > K)  
        elif(contract_type == CONTRACT_PUT_OPTION): 
            return -1 * (S < K) 
        else: 
            raise ValueError(f"Invalid contract type {contract_type}") 
  
    # Options not at expiry are somewhere in between  
    d1, d2 = bsm_components(sigma, S, K, r, T)
  
    if contract_type == mining_constants.CONTRACT_CALL_OPTION: 
        delta = norm.cdf(d1) 
    
    elif contract_type == mining_constants.CONTRACT_PUT_OPTION: 
        delta = norm.cdf(d1) - 1 
    
    else: 
        raise ValueError(f"Invalid contract type {contract_type}") 
  
    return delta 
  
  
def bsm_gamma(sigma: float, S: float, K: float, r: float, T: float, contract_type: str) -> float: 
  
    d1, d2 = bsm_components(sigma, S, K, r, T)
   
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)) 
  
    return gamma 
  
  
def bsm_theta(sigma: float, S: float, K: float, r: float, T: float, contract_type: str) -> str: 
  
    d1, d2 = bsm_components(sigma, S, K, r, T)
  
    ## 
    if contract_type == CONTRACT_CALL_OPTION: 
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2) 
    
    elif contract_type == CONTRACT_PUT_OPTION: 
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2) 
    
    else:
        raise ValueError(f"Invalid contract type {contract_type}") 

    return theta 
  
  
def bsm_rho(sigma: float, S: float, K: float, r: float, T: float, contract_type: str) -> str:
    
    ## 
    d1, d2 = bsm_components(T, S, K, r, sigma) 
  
    if contract_type == CONTRACT_CALL_OPTION: 
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) 
    
    elif contract_type == CONTRACT_PUT_OPTION:
        rho = -(K * T * np.exp(-r * T)) * norm.cdf(-d2) 
    else:
        raise ValueError(f"Invalid contract type {contract_type}")
  
    return rho  
  
  
def bsm_vega(sigma: float, S: float, K: float, T: float, r: float, contract_type: str) -> float: 
  
    ## 
    d1, d2 = bsm_components(T, S, K, r, sigma) 
  
    vega = S * np.sqrt(T) * norm.pdf(d1) 
  
    return vega / 100 

# Multiprocessor. 
def compute_option_greek(fh, P_t, IV_t, t_vec, K_vec, ttm, r, ctype, n_process=8): 
    """ 
    Computes Greek Function for matrix input using Python Multiprocess functionality.
    
    Input:
        fh     - function handle.
        P_t    - [T x N] price matrix 
        IV_t   - [T x N] matrix of delta hedging values 
        t_vec  - [T x 1] vector of timestamp for price/sigma 
        K_vec  - strike [1 x N] 
        ttm    - ttm of the option 
        r      - risk-free-rate [scalar] 
        ctype  - contract type: {CONTRACT_CALL_OPTION | CONTRACT_PUT_OPTION}
    """ 

    with multiprocessing.Pool(n_process) as pool: 

        ## get size of montecarlos models 
        T, N = np.shape(P_t) 

        ## allocate matrix to individual monte-carlo tests. 
        ## 
        ## P_t[t, :]    - 1 x N vector of prices for a single timeslice. 
        ## IV_t[t, :]   - 1 x N vector of sigma_iv values. 
        ## strike       - 1 x N vector for each MC run (time indpendent.) 
        ## ttm          - time to maturity  [scalar] 
        ## r            - 1 [scalar] 
        ## ct           - {CONTRACT_CALL_OPTION | CONTRACT_PUT_OPTION} 
        ## 
        
        args = [(IV_t[k, :], P_t[k, :], K_vec, ttm - t_vec[k], r, ctype) for k in range(len(t_vec)) ] 

        ## star-map 
        results = pool.starmap(fh, args) 

        ## need to reshape the array of arrays. 
        GREEK_t = np.concatenate(results).reshape(np.shape(P_t)) 

        return GREEK_t

