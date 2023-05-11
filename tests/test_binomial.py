
import sys
sys.path.append("../lib")

import binomial

import numpy as np


############################################################ 
# Dollar Price Test 
############################################################ 
def test_call_pricer():

    sigma = 0.5
    S = 10
    K = 10
    r = 0.02
    n = 100
    T = 1

    call_price, call_delta = binomial.usd_call_price(sigma, S, K, r, n, T)

    assert round(call_price, 3) == 2.050
    assert round(call_delta, 3) == 0.614
      

    call_price, call_delta = binomial.btc_call_price(sigma, S, K, r, n, T)

    assert round(call_price, 3) == 0.113
    assert round(call_delta, 3) == 0.029

