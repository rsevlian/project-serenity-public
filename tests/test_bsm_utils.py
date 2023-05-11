import sys
sys.path.append("../lib")

import mining_constants
import bsm_utils
import numpy as np

def test_close_to_expiry(): 
    
    # Test itm price 
    # Expiring 
    assert bsm_utils.bsm_price(sigma=0.5, S=9, K=5, r=0.05, T=0, contract_type=mining_constants.CONTRACT_CALL_OPTION) == 4 
    assert bsm_utils.bsm_price(sigma=0.5, S=5, K=9, r=0.05, T=0, contract_type=mining_constants.CONTRACT_PUT_OPTION) == 4 

    # Close to expiry 
    assert bsm_utils.bsm_price(sigma=0.5, S=9, K=5, r=0.05, T=0.01, contract_type=mining_constants.CONTRACT_CALL_OPTION) > 4 
    assert bsm_utils.bsm_price(sigma=0.5, S=5, K=9, r=0.05, T=0.01, contract_type=mining_constants.CONTRACT_PUT_OPTION) < 4 
  
    # Test otm 
    # Expiring 
    assert bsm_utils.bsm_price(sigma=0.5, S=4, K=5, r=0.05, T=0, contract_type=mining_constants.CONTRACT_CALL_OPTION) == 0 
    assert bsm_utils.bsm_price(sigma=0.5, S=5, K=4, r=0.05, T=0, contract_type=mining_constants.CONTRACT_PUT_OPTION) == 0 
    
    # Close to expiry 
    assert bsm_utils.bsm_price(sigma=1.5, S=4, K=5, r=0.05, T=0.0001, contract_type=mining_constants.CONTRACT_CALL_OPTION) > 0 
    assert bsm_utils.bsm_price(sigma=1.5, S=5, K=4, r=0.05, T=0.0001, contract_type=mining_constants.CONTRACT_PUT_OPTION) > 0 
  
    # Spot Checked against: http://www.option-price.com/index.php
    c_price = bsm_utils.bsm_call_price(sigma=0.5, S=10, K=10, r=0.02, T=1)
    p_price = bsm_utils.bsm_put_price(sigma=0.5, S=10, K=10, r=0.02, T=1)

    # Call Price:  2.0551190765500222
    # Call Price:  1.8571058096175763
    assert round(c_price, 3) == 2.055
    assert round(p_price, 3) == 1.857  
  

def test_delta_calls(): 

    # Spot Checked against: http://www.option-price.com/index.php
    c_delta = bsm_utils.bsm_delta(sigma=0.5, S=10, K=10, r=0.02, T=1, contract_type=mining_constants.CONTRACT_CALL_OPTION)
    p_delta = bsm_utils.bsm_delta(sigma=0.5, S=10, K=10, r=0.02, T=1, contract_type=mining_constants.CONTRACT_PUT_OPTION)

    assert round(c_delta, 3) == 0.614
    assert round(p_delta, 3) == -0.386

def test_vectorized_delta_calc():

    sigma = 0.5
    S     = 10
    K     = 10
    T     = 1
    P_t   = S * np.ones((5, 10))
    IV_t  = sigma * np.ones((5, 10))
    fh    = bsm_utils.bsm_delta
    K_vec = K * np.ones((1, 10))
    t_vec = np.linspace(0, T, 5)
    r     = 0.03

    D_t = bsm_utils.compute_option_greek(fh, P_t, IV_t, t_vec, K_vec, T, r, ctype = mining_constants.CONTRACT_CALL_OPTION, n_process=8)
    d1  = bsm_utils.bsm_delta(sigma, S, K, T - t_vec, r, mining_constants.CONTRACT_CALL_OPTION)

    d2 = D_t[:, 0]

    assert (d1[~np.isnan(d1)] == d2[~np.isnan(d2)]).sum() == 4
    




# def strictly_increasing(series): 
#     return all(x < y for x, y in zip(series, series[1:])) 
# 
# 
# def strictly_decreasing(series): 
#     return all(x > y for x, y in zip(series, series[1:])) 
#  
# 
# def non_increasing(series): 
#     return all(x >= y for x, y in zip(series, series[1:])) 
#
# def non_decreasing(series): 
#     return all(x <= y for x, y in zip(series, series[1:])) 
#  
# def monotonic(series): 
#     return non_increasing(series) or non_decreasing(series)   
#  
# def strictly_monotonic(series): 
#     return strictly_decreasing(series) or strictly_increasing(series) 
# 
# def avg_consec(series): 
#     return np.array([(a + b) / 2 for a, b in zip(series, series[1:])]) 
  
  
############################################################ 
# Dollar Price Test 
############################################################ 
# def test_pass():
#     assert True
  

# def test_sweep_vol():
  
#     # Increasing vol should increase option price 
#     v = np.linspace(0.1, 4, 10) 

#     # bsm_call_price(sigma: float, S: float, K: float, r: float, T: float)
#     p_t = bsm_utils.bsm_price(sigma=v, S=5, K=5, r=0.05, T=1, contract_type=bsm_utils.CONTRACT_CALL_OPTION) 
    
#     assert strictly_increasing(p_t)
  

# def test_sweep_ir():
  
#     # Increasing interest rate should increase option price 
#     ir = np.linspace(0.01, 0.1, 10) 

#     # bsm_call_price(sigma: float, S: float, K: float, r: float, T: float)
#     p_t = bsm_utils.bsm_price(sigma=0.5, S=5, K=5, r=ir, T=1, contract_type=bsm_utils.CONTRACT_CALL_OPTION) 
# 
#     assert strictly_increasing(p_t) 
# 
# 
# Increasing volatility 
# OTM 
# assert True
# v = np.linspace(0.1, 0.5, 10) 
# d_t = bsm_utils.bsm_delta(sigma=15, S=20, K=1.0, r=0.05, T=0.01, contract_type=bsm_utils.CONTRACT_CALL_OPTION) 
# print(d_t)
# assert strictly_increasing(d_t) 

# ITM 
# d_t = bsm_utils.bsm_delta(sigma=25, S=20, K=1.0, r=0.05, T=0.01, contract_type=bsm_utils.CONTRACT_CALL_OPTION)     
# assert strictly_decreasing(d_t) 

# Close to expiry, delta approaches 0 or 1 
# ttm = 0.1 - np.linspace(0, 0.1, 5) 

# OTM 
# d_t = [bsm_utils.bsm_delta(sigma=0.25, S=15, K=20, T=t, r=0.05, contract_type=bsm_utils.CONTRACT_CALL_OPTION) for t in ttm] 
# assert strictly_decreasing(d_t)

# # ITM 
# d_t = [bsm_utils.bsm_delta(sigma=0.25, S=20, K=20, T=t, r=0.05, contract_type=bsm_utils.CONTRACT_CALL_OPTION) for t in ttm] 
# assert strictly_increasing(d_t) 






# def test_delta_puts(): 
  
#     # Increasing volatility 
#     # OTM 
#     vol = np.linspace(0.1, 0.5, 10) 
#     d_t = bsm_utils.bsm_delta(sigma=vol, S=15, K=20, T=1.0, r=0.05, contract_type=bsm_utils.CONTRACT_PUT_OPTION) 
#     assert strictly_increasing(d_t) 
    
#     # ITM 
#     d_t = bsm_utils.bsm_delta(sigma=vol, S=25, K=20, T=1.0, r=0.05, contract_type=bsm_utils.CONTRACT_PUT_OPTION) 
#     assert strictly_decreasing(d_t) 
  
#     # Close to expiry, delta approaches 0 or 1 
#     ttm = 0.1 - np.linspace(0, 0.1, 5) 
    
#     # OTM 
#     d_t = [bsm_utils.bsm_delta(sigma=0.25, S=15, K=20, T=t, r=0.05, contract_type=bsm_utils.CONTRACT_PUT_OPTION) for t in ttm]
#     assert strictly_decreasing(d_t) 
    
#     # ITM 
#     d_t = [bsm_utils.bsm_delta(sigma=0.25, S=25, K=20, T=t, r=0.05, contract_type=bsm_utils.CONTRACT_PUT_OPTION) for t in ttm] 
#     assert strictly_increasing(d_t) 
  
  
# def test_gamma(): 
  
#     # Test gamma 
#     # Vary volatility 
#     vol = np.linspace(0.1, 1.5, 5) 
#     g_t = bsm_utils.bsm_gamma(sigma=vol, S=15, K=15, r=0.01, T=0.5) 
#     # assert strictly_decreasing(g_t) 
  
#     # Test gamma is second derivative of price series 
#     spots, step = np.linspace(90, 110, 200, retstep=True) 
#     p_t = bsm_utils.bsm_price(sigma=0.25, S=spots, K=100, r=0.05, T=1.0, contract_type=bsm_utils.CONTRACT_CALL_OPTION)
#     d_t = bsm_utils.bsm_delta(sigma=0.25, S=100, K=100, T=1.0, r=0.05, contract_type=bsm_utils.CONTRACT_CALL_OPTION)
#     g_t = bsm_utils.bsm_gamma(sigma=0.25, S=spots, K=100, T=1.0, r=0.05)
  
#     # assert np.allclose(np.diff(p_t) / step, avg_consec(d_t)) 
  

# def test_theta(): 
  
#     # Test options theta 
#     ttm_vec = 1.01 - np.linspace(0, 1, 10) 
#     thetas = [bsm_utils.bsm_theta(sigma=0.5, S=10, K=10, T=ttm, r=0.05, contract_type=bsm_utils.CONTRACT_CALL_OPTION) for ttm in ttm_vec] 
#     # assert strictly_decreasing(thetas)
  
#     thetas = [bsm_utils.bsm_theta(10, 10, ttm, 0.05, 0.5, bsm_utils.CONTRACT_PUT_OPTION) for ttm in ttm_vec] 
#     # assert strictly_decreasing(thetas)
  

# def test_rho(): 
#     pass 
  

