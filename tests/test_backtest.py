
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

import backtest


def test_monte_carlo_delta_hedger():

	import delta_hedge
	# imp.reload(delta_hedge)

	import backtest
	# imp.reload(backtest)

	mc_params = {
	    "S": 100,
	    "K": 100,
	    "T": 1/52,
	    "mc_size": 500, 
	    "sigma_iv": 0.4, 
	    "sigma_rv": 0.4,
	    "interest_rate" : 0.02,
	    "Q": 100*1000, 
	    "contract_direction": mining_constants.DIRECTION_LONG_CONTRACT, 
	    "contract_type": mining_constants.CONTRACT_CALL_OPTION, 
	    "sample_rate": mining_constants.SAMPLE_RATE_HOURLY, 
	    "risk_free_rate": 0.02,  
	    "slippage": 0.0001
	}

	np.random.seed(0)
	mcb = backtest.MonteCarloBacktest(mc_params)
	sim_df = mcb.price_model()
	sim_df.head()


	assert sim_df.hedging_cost.mean() == 218120.28103539508
	assert sim_df.agg_bsm_price.mean() == 223064.6637804188
	assert sim_df.hedging_cost.std() == 15783.200908853078