import datetime 
import data_utils 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import mining_constants 

import scipy.stats as stats 
import sys 
import os 

from monte_carlo import GBMPriceLinearHashRate
import bsm_utils
import binomial

import monte_carlo

def _simulate_machine_loan_usd(t_vec, s_t, hr_t, h_t, w_t, params): 
  
	# miner input params 
	#  - electricity_price: 
	#  - energy_consumed: 
	#  - hash_per_asic: 
	#  - asic_efficiency: 
	#  - pool_fee: 
	#  - asic_number: 
  
	# logging.info("machine_npv_simulation_v1")
	df = pd.DataFrame({"ttm" : t_vec,  "s_t" : s_t, "hr_t" : hr_t, "w_t" : w_t}) 

	df["reward"] = df["w_t"] * (params["asic_hash_rate"] / df["hr_t"]) * (1 - params["pool_fee"]) 


	## compute weekly cost of BTC production i.e. the strike price of the option. 
	weekly_cost  = params["electricity_cost"] * params["asic_energy_consumption"] * mining_constants.DAYS_IN_WEEK * mining_constants.HOURS_IN_DAY  
	df["strike"] = weekly_cost / df["reward"] 

	##  
	df["call_payoff"] = df.apply(lambda x : np.max([x.s_t - x.strike, 0]), axis = "columns") 


	## discount factor array. 
	df["discount_factor"] = np.power(1 + params["interest_rate"]/params["sample_rate"], np.arange(start=0, stop=len(df), step=1)) 

	## discount each weekly revenue value.  
	df["machine_call_payoff"] = df["call_payoff"] * df["reward"] / df["discount_factor"]  

	return df["machine_call_payoff"].sum()


# TODO: generate monte-carlo simulation.
class LoanSimulation_USD(LoanSimulation):

	def default_rate(self, start_time: float, spot_init: float, hashrate_init: float, mu_annual):

		self._sp._gbm_params["mu"] = np.power(1 + mu_annual, 1/self._params["sample_rate"]) - 1

		t_vec, dt_vec, St, HRt, wt, Ht = self._sp.simulate(
														start_time, 
														spot_init, 
														hashrate_init, 
														T = self._params["machine_duration"], 
														sample_rate = self._params["sample_rate"], 
														N = self._params["mc_size"] 
													)

		result_stats = monte_carlo.run_monte_carlo_simulation(
														_simulate_machine_npv,
						    							t_vec,
			 	                                        St, 
			 	                                       	HRt, 
			 	                                        Ht, 
									 					wt,
			 	                                        self._params,
			 	                                        n_process = 3
			 	                                     )

		npv_vals = result_stats.values
		npv_mean = np.mean(npv_vals)
		npv_stdev = np.std(npv_vals)
		
		return {
			"mean": npv_mean, 
			"stdev": npv_stdev
		}



# # TODO: generate monte-carlo simulation.
# class LoanSimulation_BTC(MachineNPV):
# 
# 	def price(self, start_time: float, spot_init: float, hashrate_init: float, mu_annual):
# 
# 		self._sp._gbm_params["mu"] = np.power(1 + mu_annual, 1/self._params["sample_rate"]) - 1
# 
# 		t_vec, dt_vec, St, HRt, wt, Ht = self._sp.simulate(
# 														start_time, 
# 														spot_init, 
# 														hashrate_init, 
# 														T = self._params["machine_duration"], 
# 														sample_rate = self._params["sample_rate"], 
# 														N = self._params["mc_size"] 
# 													)
# 
# 		result_stats = monte_carlo.run_monte_carlo_simulation(
# 														_simulate_machine_npv,
# 						    							t_vec,
# 			 	                                        St, 
# 			 	                                       	HRt, 
# 			 	                                        Ht, 
# 									 					wt,
# 			 	                                        self._params,
# 			 	                                        n_process = 3
# 			 	                                     )
# 
# 		npv_vals = result_stats.values
# 		npv_mean = np.mean(npv_vals)
# 		npv_stdev = np.std(npv_vals)
# 
# 		return {
# 			"mean": npv_mean, 
# 			"stdev": npv_stdev
# 		}



