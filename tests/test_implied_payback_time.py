import sys
sys.path.append("../lib")

import pandas as pd
import data_utils
import monte_carlo
import bsm_utils
import numpy as np

import machine_npv
import mining_constants

import implied_payback_time

def test_implied_payback_time():


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
	du = data_utils.DataUtility("")
	du._raw_network_price_df    = pd.read_csv("../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")

	du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
	du._df_feature() 
	du._aggregate_weekly_df() 

	machine_prices = data_utils.machine_market_prices("../data/machine_prices.csv")
	du._weekly_machine_price_df = du._join_machine_price(machine_prices, HR_ROLLING_AVG=30)

	weekly_df = du._weekly_network_price_df
	machine_df = du._weekly_machine_price_df
	
	## 
	sp = monte_carlo.GBMPriceLinearHashRate()
	sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

	## 
	k = len(weekly_df) - 1

	m_usd_bsm = machine_npv.MachineNPV_USD_BSM(sp, params)
	# usd_bsm   = m_usd_bsm.price(start_time, btc_close, hash_rate)

	ipt    = implied_payback_time.ImpliedPayoffTime(params, m_usd_bsm, machine_df.tail(5))
	ipt_df = ipt.compute(machine_type=mining_constants.MACHINE_M20, alpha=0.75, beta=0.5, MAX_DURATION=5)

	e = ipt_df.ipb_ttm.values - [1.11969112, 1.1003861 , 0.94594595, 1.31274131, 1.58301158]
	assert round(sum(abs(e)), 3) == 0
