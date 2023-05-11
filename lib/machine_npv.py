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

def _simulate_machine_npv(t_vec, s_t, hr_t, h_t, w_t, params): 
  
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
	if params.get("BTC_DENOMINATED"):
		df["call_payoff"] = df.apply(lambda x : np.max([1 - x.strike / x.s_t, 0]), axis = "columns") 
	else:
		df["call_payoff"] = df.apply(lambda x : np.max([x.s_t - x.strike, 0]), axis = "columns") 

	## discount factor array. 
	df["discount_factor"] = np.power(1 + params["interest_rate"]/params["sample_rate"], np.arange(start=0, stop=len(df), step=1)) 

	## discount each weekly revenue value.  
	df["machine_call_payoff"] = df["call_payoff"] * df["reward"] / df["discount_factor"]  

	return df["machine_call_payoff"].sum()


class MCDefaultRate:

	def __init__(self, sp: GBMPriceLinearHashRate, params: dict):

		self._sp = sp
		self._params = params
	
	def default_rate_usd(self, start_time: float, spot_init: float, hashrate_init: float, mu_annual:float, loan_value: float):

		# self._sp._gbm_params["mu"] = np.power(1 + mu_annual, 1/self._params["sample_rate"]) - 1

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

		# npv_vals = result_stats.values
		# npv_mean = np.mean(npv_vals)
		# npv_stdev = np.std(npv_vals)
		
		return (result_stats < loan_value).mean().values[0]
		

class MachineNPV:

	def __init__(self, sp: GBMPriceLinearHashRate, params: dict):

		self._sp = sp
		self._params = params
		self._option_df = None

	def _setup(self, start_time: float, spot_init: float, hashrate_init: float, lambda_C=1):

		# generate a model hashrate & reward and ttm vector. 
		t_vec, dt_vec, St, HRt, wt, _ = self._sp.simulate( 
		                                                start_time,  
		                                                spot_init,  
		                                                hashrate_init,   
		                                                T = self._params["machine_duration"],  
		                                                sample_rate = self._params["sample_rate"],  
		                                                N = 2,  # nominal number to generate t_vec, HRt, wt.
		                                                lambda_C=lambda_C                         
		                                            ) 

		# generate the dataframe 
		option_df = pd.DataFrame( 
		                        data={ 
		                             "ttm" : t_vec,  
		                             "ttm_dt" : dt_vec,  
		                             "spot" : spot_init,  
		                             "network_hashrate" : HRt,  
		                             "network_reward": wt 
		                            } 
		                        ) 

		# 
		option_df["reward"] = option_df["network_reward"] * (self._params["asic_hash_rate"] / option_df["network_hashrate"]) 

		# option volatility => making it annualized.
		option_df["sigma"] = self._sp._gbm_params["sigma"] * np.sqrt(self._params["sample_rate"]) 

		# compute weekly cost of BTC production i.e. the strike price of the option. 
		weekly_cost = self._params["electricity_cost"] * self._params["asic_energy_consumption"] * \
							mining_constants.DAYS_IN_WEEK * mining_constants.HOURS_IN_DAY 
		 
		option_df["weekly_cost"] = weekly_cost  
		option_df["strike"] = option_df["weekly_cost"] / option_df["reward"] 

		self._option_df = option_df

# TODO: generate monte-carlo simulation.
class MachineNPV_MCSim(MachineNPV):

	def price(self, start_time: float, spot_init: float, hashrate_init: float, mu_annual):

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


class MachineNPV_USD_BSM(MachineNPV):

	def price(self, start_time: float, spot_init: float, hashrate_init: float):

	    # """ 
	    # 	BSM USD machine pricing model.
		#     Input:
		#         start_time    - 
		# 		spot_init     -
		# 		hashrate_init -
		
		#     Output:
		# 		output_df     - 
	    # """
		 
		self._setup(start_time, spot_init, hashrate_init)

		## price each weekly reward as a call option. 
		self._option_df["call_price"] = self._option_df.apply(lambda x: 
					bsm_utils.bsm_call_price(x.sigma, x.spot, x.strike, self._params["interest_rate"], x.ttm), 
					axis="columns"
				) 

		## compute the instantatious delta.
		self._option_df["call_delta"] = self._option_df.apply(
							lambda x: bsm_utils.bsm_delta(
										x.spot, 
										x.sigma, 
										x.strike, 
										self._params["interest_rate"], 
										x.ttm, 
										contract_type=mining_constants.CONTRACT_CALL_OPTION
									), 
							axis="columns"
						)

		self._option_df["reward_value"] = self._option_df["call_price"] * self._option_df["reward"]

		## the first option should not be delivered since no work was done
		self._option_df.reward_value[self._option_df.ttm == 0.0] = 0

		return self._option_df["reward_value"].sum()

class MachineNPV_USD_Bin(MachineNPV):

	def price(self, start_time: float, spot_init: float, hashrate_init: float, n: int):

		## 
		self._setup(start_time, spot_init, hashrate_init)

		# btc_call_price(sigma: float, S: float, K: float, r: float, n: int, T: float)
		self._option_df["call_price_delta"] = self._option_df.apply(
		        lambda x: binomial.usd_call_price(x.sigma, x.spot, x.strike, self._params["interest_rate"], n, x.ttm),  
		        axis="columns"
		    ) 

		self._option_df["call_price"] = self._option_df.apply(lambda x: x.call_price_delta[0], axis="columns") 
		self._option_df["call_delta"] = self._option_df.apply(lambda x: x.call_price_delta[1], axis="columns") 
		self._option_df.drop(["call_price_delta"], inplace=True, axis="columns") 

		## 
		self._option_df["reward_value"] = self._option_df["call_price"] * self._option_df["reward"]

		return self._option_df["reward_value"].sum()

class MachineNPV_BTC_Bin(MachineNPV):

	def price(self, start_time: float, spot_init: float, hashrate_init: float, n: int):

		# """ 
		# Binomial BTC Machine Pricing Model.
		# 
		# Input:
		# 	start_time    - 
		# 	spot_init     -
		# 	hashrate_init -
		# 
		# Output:
		# 	output_df     - 
		# """

		self._setup(start_time, spot_init, hashrate_init)

		n = 100 

		self._option_df["call_price_delta"] = self._option_df.apply(
				lambda x: binomial.btc_call_price(x.sigma, x.spot, x.strike, self._params["interest_rate"], n, x.ttm),  
				axis="columns"
			) 

		self._option_df["call_price"] = self._option_df.apply(lambda x: x.call_price_delta[0], axis="columns") 
		self._option_df["call_delta"] = self._option_df.apply(lambda x: x.call_price_delta[1], axis="columns") 
		self._option_df.drop(["call_price_delta"], inplace=True, axis="columns") 

		## 
		self._option_df["reward_value"] = self._option_df["call_price"] * self._option_df["reward"]

		return self._option_df["reward_value"].sum()

