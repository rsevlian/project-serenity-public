import datetime
import data_utils
import pandas as pd 
import logging 
import numpy as np 
import statsmodels.api as sm 
import mining_constants    
import multiprocessing

np.seterr(divide = 'ignore') 

def date2datetime(dt):
    return datetime.datetime(dt.year, dt.month, dt.day, 0, 0)

def run_monte_carlo_simulation(simulator, t_vec, St, HRt, Ht, Wt, params, n_process=8): 
  
    ## print("simulator = ", simulator) 
    ## 
    with multiprocessing.Pool(n_process) as pool: 

        ## get size of montecarlos models 
        mc_size = np.shape(Ht)[1] 

        ## allocate matrix to individual monte-carlo trajectories. 
        ## t_vec - T x 1 vector of time to maturities. 
        ## St    - T x N matrix of spot prices. 
        ## HRt   - T x 1 vector of hashrate growth. 
        ## Ht    - T x N matrix of hash index trajectories.   
        args = [(t_vec, St[:, k], HRt, Ht[:, k], Wt, params) for k in range(mc_size)] 

        ## star-map 
        results = pool.starmap(simulator, args) 

        return pd.DataFrame(results)

def gbm_simulate(S: float, T: float, r: float, sigma: float, steps: int, N: int) -> tuple[float, np.ndarray, np.ndarray]: 
    """ 
    Generate GBM MonteCarlo Simulation.
    
    Inputs 
        S     : Current stock Price 
        T     : Time to maturity 1 year = 1, 1 months = 1/12 
        r     : risk free interest rate  
        sigma : sample volatility 
        steps : number of time steps
        N     : number of monte-carlo trajectories
  
    Output 
        t     : vector of time [0, T] via n steps.
        St    : N x steps matrix of generated price paths.
        Rt    : N x steps matrix of generated GBM returns. 
    """ 
  
    # mean 
    mu_gbm = (r - sigma**2/2) 
  
    # zero_row + sample returns 
    zr = np.zeros((1, N)) 
  
    # take samples from normal distribution. 
    Rt = np.random.normal(loc=mu_gbm, scale = sigma, size=(steps-1, N)) 
  
    # append the zero row to the sample returns. 
    Rt = np.append(zr, Rt, axis=0) 
  
    # S(t) = S0 x exp( sum_k(r_k) ) 
    logS = np.log(S) + np.cumsum(Rt, axis = 0) 
    St = np.exp(logS)
  
    # vector of timestamps. 
    t = np.linspace(0, T, steps) 
  
    return t, St, Rt 
  
class GBMPriceLinearHashRate(): 
  
    def init(self): 
  
        # dataframe used to fit the hashrate growth. 
        self._lm = None 
        self._lm_data = None

        # gbm params. 
        self._gbm_params = None 
  
    def _fit_gbm_series(self, price_sr: pd.Series) -> dict:

        """
            Input:
                price_sr - index: time, value: weekly close.
            Output:
                mu       - avg returns 
                stdev    - return volatility
        """

        log_returns = np.log(price_sr / price_sr.shift(1)) 

        self._gbm_params = { 
            "mu": log_returns.mean(), 
            "sigma": log_returns.std() 
        } 

    def _fit_linear_hashrate(self, hashrate_df: pd.DataFrame):

        """
            Helper function used to compute simple regression:
            Input:
        """
        
        # filter weekly df to support 
        # hr1 = hashrate_df[
        #         ((hashrate_df.index > mining_constants.LM_S1) & (hashrate_df.index < mining_constants.LM_E1)) |  
        #         ((hashrate_df.index > mining_constants.LM_S2) & (hashrate_df.index < hashrate_df.index[-1]))
        #     ] 
        hr1 = hashrate_df
        

        # normalized time. 
        Y = hr1.values
        X = np.array([(hr1.index[k] - hr1.index[0]).days/365 for k in range(len(hr1))])
        
        X = sm.add_constant(X) 
        lm = sm.OLS(Y, X).fit() 
      
        self._lm = lm 
        self._lm_data = {"x": X, "y": Y} 

    def fit(self, weekly_df: pd.DataFrame, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY): 
  
        # set sample rate. 
        self._fit_sample_rate = sample_rate 

        # simple GBM params: mean/stdev of returns. 
        self._fit_gbm_series(weekly_df.close)
  
        # fit hashrate 
        self._fit_linear_hashrate(weekly_df[["hashrate"]]) 
  
    def _hashrate_forecast(self, hashrate_init, ttm_vec, C=1): 
        
        LM_CONST_IDX = 0
        LM_SLOPE_IDX = 1

        # extrapolate hash-rate trend. 
        # generate feature: [1, t] for constant. 
        Xpred = pd.DataFrame({"year" : ttm_vec}) 
        m = self._lm.params[LM_SLOPE_IDX]
        
    
        self._lm.params[LM_SLOPE_IDX]   = m * C
        self._lm.params[LM_CONST_IDX] = hashrate_init 
        HRt = self._lm.predict(sm.add_constant(Xpred)).values 

        self._lm.params[LM_SLOPE_IDX]   = m 

        return HRt 
  
    def simulate(self, start_time: datetime.datetime, spot_init: float, hashrate_init: float, T: float, sample_rate: float, N: int, lambda_C=1): 
        
        """
        Simulate Weekly Price, Hashrate, Mining Output: 
        Input: 
            start_time      - simulation start time.
            spot_init       - spot price value
            hashrate_init   - hashrate value
            T               - simulation horizon.
            sample_rate     - simulation sample rate.
            N               - number of monte-carlo simulations. 
        Output:
            ttm_vec - [1 x M] vector of ttms [0, T]
            dt_vec  - [1 x M] vector of datetime objects.
            St      - [M x N] matrix of GBM simulation paths.
            HRt     - [1 x M] hashrate projection
            Qt      - [1 x M] mining output quantity
            Ht      - [M x N] simulated hashindex paths.
        """
        if not ((sample_rate == mining_constants.SAMPLE_RATE_WEEKLY) | (sample_rate == mining_constants.SAMPLE_RATE_HOURLY)) :
            raise ValueException("sample_rate: {} not allowed.".format(sample_rate))

        steps = int(sample_rate * T) 
        mu    = self._gbm_params["mu"] * (self._fit_sample_rate / sample_rate) 
        sigma = self._gbm_params["sigma"] * np.sqrt(self._fit_sample_rate / sample_rate) 

        # logging.info( 
        #     "simulate: T = {} sample_rate = {} => steps = {} vol-scale({}/{}) vol {} => {} mu {} = > {}"\
        #     .format(
        #             T, 
        #             sample_rate, 
        #             steps, 
        #             self._fit_sample_rate, 
        #             sample_rate, 
        #             self._gbm_params["sigma"], 
        #             sigma, 
        #             self._gbm_params["mu"], 
        #             mu
        #         ) 
        #     ) 

        ## generate GBM simulation output.
        ttm_vec, St, Rt = gbm_simulate(spot_init, T, mu, sigma, steps, N) 
        
        # print(len(ttm_vec), St.shape, Rt.shape)

        ## Generate TTM datetimes. 
        ttm_dt_vec = [date2datetime(start_time) + pd.Timedelta(hours = int(365 * 24 * dt)) for dt in ttm_vec] 
        # print(ttm_vec[0:10])
        # print(ttm_dt_vec[0:10])
        
        ##
        HRt = self._hashrate_forecast(hashrate_init, ttm_vec, lambda_C) 
  
        ## MC generated Hash-Index. 
        wt_daily = np.array([data_utils.daily_btc_reward(start_time, dt) for dt in ttm_vec]) 
  
        ## samples per day
        C = (mining_constants.DAYS_IN_YEAR / sample_rate)  
        
        ##
        # logging.info("simulate: daily_reward x (365 / {}) = daily_reward x ({})".format(sample_rate, C)) 
        # print(C, wt_daily.shape, St.shape, HRt.shape)
        
        Ht = C * (wt_daily * St.T / HRt ).T  
  
        return ttm_vec, ttm_dt_vec, St, HRt, C * wt_daily, Ht


