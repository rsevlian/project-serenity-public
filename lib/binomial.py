import numpy
import math  

    
def lattice_params(T: float, n: float, sigma: float, r: float) -> float: 
  
    """
        Generate lattice parameters:
        input:
            T     - time to maturity (years)
            n     - discretization size (int)
            sigma - annualized volatilty (100 %)
            r     - risk free interest rate (100 %)
        
        output:
            dt - discrete time
            u  - size of up move
            d  - size of down move (d=1/u)
            p  - probability of up move
    """  

    dt = T / n 
    u = numpy.exp(sigma * numpy.sqrt(dt)) 
    d = 1 / u 
    p = (numpy.exp(r * dt) - d) / (u - d) 
  
    return dt, u, d, p 


def generate_price_lattice(S: float, u: float, d: float, n: int) -> numpy.ndarray: 
  
    """
        Generates Binomial Pricing Lattice:
        input: 
            S - current spot price
            u - size of up move
            d - size of down move (d=1/u)
            n - discretization size (int)
        output:
            price_lattice - upper triangular numpy.ndarray
    """
    price_lattice = numpy.zeros((n, n)) 
    for i in range(0, n, 1): 
        for j in range(0, n, 1): 
            price_lattice[i, j] = S * numpy.power(u, j - i) * numpy.power(d, i) 
  
    return numpy.triu(price_lattice) 
  
def generate_usd_call_lattice(price_lattice: numpy.ndarray, K: float, r: float, p: float, dt: float) -> numpy.ndarray:
    """
        Generate Call Price Lattice:
        
        input: 
            price_lattice - upper triangular matrix of prices.
            K             - strike price
            r             - risk free rate
            p             - probability of up move
            dt            - time discretization
        
        output:
            call_lattice - upper triangular numpy.ndarray
    """

    n = price_lattice.shape[0] 
    Cm = numpy.zeros((n, n)) 
  
    # Initialize Call Option Price. 
    j = n-1 
    for i in range(n): 
        Cm[i, j] = max(price_lattice[i, j] - K, 0) 
  
    for j in range(n-2, -1, -1): 
        for i in range(0, n-1, 1): 
            if j >= i: 
                 Cu   = Cm[i, j + 1] 
                 Cd   = Cm[i + 1, j + 1] 
                 Cnew = numpy.exp(- r * dt) * (p * Cu + (1 - p) * Cd) 
  
                 Cm[i, j] = Cnew 
  
    return Cm
  
def generate_btc_call_lattice(price_lattice: numpy.ndarray, K: float, r: float, p: float, dt: float) -> numpy.ndarray: 
  
    n = price_lattice.shape[0] 
  
    Cm = numpy.zeros((n, n)) 
  
    j = n-1 
    for i in range(n): 
        Cm[i, j] = max(1 - K/price_lattice[i, j], 0) 
  
    for j in range(n-2, -1, -1): 
        for i in range(0, n-1, 1): 
            if j >= i: 
                Cu   = Cm[i, j+1] 
                Cd   = Cm[i+1, j+1] 
                Cnew = numpy.exp( - r * dt ) * ( p * Cu + (1 - p) * Cd) 
  
                Cm[i, j] = Cnew 
  
    return Cm 
  
  
  
def usd_call_price(sigma: float, S: float, K: float, r: float, n: int, T: float) -> float: 
    
    ## compute parameters of the lattice. 
    dt, u, d, p = lattice_params(T, n, sigma, r)
    
    ## compute pricing lattice. 
    price_lattice = generate_price_lattice(S, u, d, n + 1) 
  
    ## compute call-option lattice. 
    call_lattice  = generate_usd_call_lattice(price_lattice, K, r, p, dt) 
  
    ## (C(t+1, +) - C(t+1, -)) / (S(t+1, +) - S(t+1, -)) 
    delta = (call_lattice[0, 1] - call_lattice[1, 1]) / (price_lattice[0, 1] - price_lattice[1, 1]) 
  
    return call_lattice[0, 0], delta 
  

def btc_call_price(sigma: float, S: float, K: float, r: float, n: int, T: float) -> float: 
  
    ## compute parameters of the lattice. 
    dt, u, d, p = lattice_params(T, n, sigma, r) 
  
    ## compute pricing lattice. 
    price_lattice = generate_price_lattice(S, u, d, n + 1) 
  
    ## compute call-option lattice. 
    call_lattice  = generate_btc_call_lattice(price_lattice, K, r, p, dt) 
  
    ## (C(t+1, +) - C(t+1, -)) / (S(t+1, +) - S(t+1, -)) 
    delta = (call_lattice[0, 1] - call_lattice[1, 1]) / (price_lattice[0, 1] - price_lattice[1, 1]) 
    
    return call_lattice[0, 0], delta

