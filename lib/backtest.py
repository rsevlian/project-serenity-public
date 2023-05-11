import sys, os 
home = os.environ["HOME"] 

import mining_constants  
import bsm_utils 
import multiprocessing 
import numpy as np 
import numpy.lib.stride_tricks as st 
import pandas as pd 
import bsm_utils 
import logging 
import delta_hedge 
import mining_constants 
import time 
import monte_carlo 



class MonteCarloBacktest():
	"""
		Generats Monte Carlo Model backtest.
		Input: 
			mc_param [dictionary]:
				S   		       - spot price
	    		K     		       - strike
	    		T     			   - time to maturity
	    		mc_size 		   - monte-carlo size.
	    		sigma_iv 		   - quoting volatility
	    		sigma_rv		   - simulation path volatility
	    		contract_quantity 
	    		contract_direction - {DIRECTION_LONG_CONTRACT | DIRECTION_LONG_CONTRACT}
	    		contract_type      - {CONTRACT_CALL_OPTION | CONTRACT_CALL_OPTION} 
	    		asset_per_contract - number of underlying shares per contract.
	    		sample_rate        - simulation sample rate.
	    		risk_free_rate     - risk free interest rate.
	    		slippage           - 0.0001
	    Output:
	"""
	def __init__(self, mc_params):

		self._params = mc_params 

	def _position_delta(self, t_vec, PRICE_t, SIGMA_IV_t, strike_vec):

		## Compute Delta_T of position 
		DELTA_t = bsm_utils.compute_option_greek( 
								bsm_utils.bsm_delta, 
								PRICE_t, 
								SIGMA_IV_t, 
								t_vec, 
								strike_vec, 
								ttm=self._params.get("T"), 
								r=self._params.get("risk_free_rate"), 
								ctype=self._params.get("contract_type"), 
							)	
		
		if self._params.get("contract_direction") == mining_constants.DIRECTION_LONG_CONTRACT:
			return DELTA_t 
		else:
			return - DELTA_t 

	def price_model(self):

		## Compute the number of steps. 
		logging.info("""mc_params = {}""".format(self._params)) 
	
		## set sample rates information.
		sr = self._params.get("sample_rate")
		ttm = self._params.get("T")
		steps = int(np.ceil(sr * ttm)) + 1 
		
		## generate price simulation.
		t_vec, PRICE_t, _ = monte_carlo.gbm_simulate(
								S = self._params.get("S"),
								T = self._params.get("T"),
								r = 0,
								sigma =  self._params.get("sigma_rv") / np.sqrt(sr),
								steps = steps,
								N = self._params.get("mc_size")
							)

		## N x T matrix of prices. 
		SLIPPAGE_t = self._params.get("slippage") * np.ones(PRICE_t.shape) 

		## 1 x N vector of strikes. 
		K_vec = self._params.get("K") * np.ones((PRICE_t.shape[1])) 

		## generate matrix of SIGMA_IV_t. 
		SIGMA_IV_t = self._params.get("sigma_iv") * np.ones(PRICE_t.shape) 

		## Compute Delta_T of position 
		DELTA_t = -1 * self._position_delta(t_vec, PRICE_t, SIGMA_IV_t, K_vec)

		## 
		quote_iv_vec = self._params.get("sigma_iv") * np.ones(PRICE_t.shape[1]) 


		# return t_vec, PRICE_t, DELTA_t, SLIPPAGE_t
		self._params["sigma"] = self._params.get("sigma_iv")
		sim_df = delta_hedge.run_delta_hedge( 
		                            delta_hedge.simulate_delta_hedging,
		                            t_vec, 
									quote_iv_vec,
		                            PRICE_t, 
		                            DELTA_t, 
		                            SLIPPAGE_t,
		                            self._params, 
		                            n_process=4 
		                        ) 
		return sim_df
		
