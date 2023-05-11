import sys
sys.path.append("../lib")

import pandas as pd
import data_utils
import monte_carlo
import bsm_utils
import numpy as np

import machine_npv
import mining_constants



def test_machine_npv():

	machine_params   = mining_constants.machine_params()
	machine_duration = 1.2
	ky = "S19J"
	params = {
	    "asic_energy_consumption" : machine_params[ky]["energy_consumption"],
	    "asic_hash_rate"          : machine_params[ky]["hash_rate"],
	    "electricity_cost"        : 0.05,
	    "machine_duration"        : machine_duration,
	    "sample_rate"             : mining_constants.SAMPLE_RATE_WEEKLY,
	    "mc_size"                 : 1000,
	    "interest_rate"           : 0.05,
	    "asic_number"             : 1,
	    "asic_efficiency"         : 1.0,
	    "pool_fee"                : 0.005,
	    "analysis_denomination"   : mining_constants.ASSET_USD
	}

	##
	du = data_utils.DataUtility("cc_key")
	du._raw_network_price_df = pd.read_csv("../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")
	du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
	du._df_feature() 
	du._aggregate_weekly_df() 

	daily_df  = du._daily_network_price_df
	weekly_df = du._weekly_network_price_df

	## 
	sp = monte_carlo.GBMPriceLinearHashRate()
	sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

	## 
	k = len(weekly_df) - 1

	start_time = weekly_df.reset_index().loc[k, "time"]
	hash_rate  = weekly_df.reset_index().loc[k, "hashrate"]
	hash_index = weekly_df.reset_index().loc[k, "hash_index"]
	btc_close  = weekly_df.reset_index().loc[k, "close"]

	## 
	m_usd_bsm = machine_npv.MachineNPV_USD_BSM(sp, params)
	usd_bsm   = m_usd_bsm.price(start_time, btc_close, hash_rate)

	m_usd_bin = machine_npv.MachineNPV_USD_Bin(sp, params)
	usd_bin   = m_usd_bin.price(start_time, btc_close, hash_rate, 100)

	m_btc_bin = machine_npv.MachineNPV_BTC_Bin(sp, params)
	btc_bin   = m_btc_bin.price(start_time, btc_close, hash_rate, 100)

	# ## 1.2 years of S19J pro at last days data.
	assert usd_bsm == 1027.4283155130154 
	assert usd_bin == 1027.3673872916377
	assert btc_bin == 0.026851440901887545
