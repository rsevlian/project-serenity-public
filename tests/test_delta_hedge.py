import sys
sys.path.append("../lib")

import bsm_utils
import numpy as np
import pandas as pd
import mining_constants

import delta_hedge

def test_delta_hedge_ITM():
	
	S  = 49
	K  = 50 
	r  = 0.05
	s  = 0.20
	T  = 0.3846
	mu = 0.13

	df_1 = pd.read_csv("../data/delta_hedge_ex_ITM.csv")

	t          = df_1.week.values / 52
	price_t    = df_1.spot.values
	sigma_iv_t = np.array(len(t) * [s])
	delta_t    = df_1.delta.values
	slippage_t = np.array(len(t) * [0])


	params = {
	    "sigma" : s,
	    "K"     : K,
	    "T"     : T,
	    "Q"     : 100000,
	    "interest_rate"      : 0.05,
	    "sample_rate"        : mining_constants.SAMPLE_RATE_WEEKLY,
	    "contract_type"      : mining_constants.CONTRACT_CALL_OPTION,
	    "contract_direction" : mining_constants.DIRECTION_SHORT_CONTRACT
	}

	metrics = delta_hedge.simulate_delta_hedging(t, s, price_t, delta_t, slippage_t, params)

	assert metrics['ITM'] == True
	assert metrics['Final Spot'] == 57.25
	assert metrics['K'] == 50
	assert metrics['purchase_cost'] == 5163087.486850535
	# assert metrics['purchase_cost_no_loss'] == -5163087.486850535
	assert metrics['hedging_cost'] == 258322.67504952103
	# assert metrics['hedging_cost_no_loss'] == -258322.67504952103
	assert metrics['expiry_cashflow'] == 4904764.811801014
	assert metrics['bsm_price'] == 2.3547394142893583
	assert metrics['agg_bsm_price'] == 235473.94142893585
	# assert round(sum(abs(df_1.shares_purchased.values -  df_2.purchase_quantity.values))) == 0

def test_delta_hedge_OTM():
	
	S  = 49
	K  = 50 
	r  = 0.05
	s  = 0.20
	T  = 0.3846
	mu = 0.13

	df_1 = pd.read_csv("../data/delta_hedge_ex_OTM.csv")

	t          = df_1.week.values / 52
	price_t    = df_1.spot.values
	sigma_iv_t = np.array(len(t) * [s])
	delta_t    = df_1.delta.values
	slippage_t = np.array(len(t) * [0])

	params = {
	    "sigma" : s,
	    "K"     : K,
	    "T"     : T,
	    "Q"     : 100000,
	    "interest_rate"      : 0.05,
	    "sample_rate"        : mining_constants.SAMPLE_RATE_WEEKLY,
	    "contract_type"      : mining_constants.CONTRACT_CALL_OPTION,
	    "contract_direction" : mining_constants.DIRECTION_SHORT_CONTRACT
	}

	metrics = delta_hedge.simulate_delta_hedging(t, s, price_t, delta_t, slippage_t, params)

	assert metrics['ITM'] == False
	assert metrics['Final Spot'] == 48.12
	assert metrics['K'] == 50
	assert metrics['purchase_cost'] == 251455.11503068352
	assert metrics['hedging_cost'] == 251455.11503068352
	assert metrics['expiry_cashflow'] == 0.0
	assert metrics['bsm_price'] == 2.3547394142893583
	assert metrics['agg_bsm_price'] == 235473.94142893585