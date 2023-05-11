import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import logging
import requests
import datetime as dt
import pylab

import scipy.stats as stats
from matplotlib import pyplot

import numpy as np

import warnings
warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.INFO)

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import os, sys

home = os.environ["HOME"]
sys.path.append(f"{home}/Desktop/project-ursa/lib")

import imp



import data_utils
import monte_carlo
import machine_npv
import implied_payback_time
import mining_constants
import delta_hedge
import machine_replication
import analysis

imp.reload(data_utils)
imp.reload(monte_carlo)
imp.reload(machine_npv)
imp.reload(mining_constants)
imp.reload(implied_payback_time)
imp.reload(delta_hedge)
imp.reload(machine_replication)
imp.reload(analysis)



##########################################################################################
## 
##########################################################################################
du = data_utils.DataUtility("")
du._raw_network_price_df    = pd.read_csv("../../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")
du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
du._df_feature() 
du._aggregate_weekly_df() 

machine_prices = data_utils.machine_market_prices("../../data/machine_prices.csv")
du._weekly_machine_price_df = du._join_machine_price(machine_prices, HR_ROLLING_AVG=30)

weekly_df = du._weekly_network_price_df
machine_df = du._weekly_machine_price_df


##########################################################################################
## 
##########################################################################################
imp.reload(monte_carlo)
machine_params   = mining_constants.machine_params()
machine_duration = 1.2
ky = mining_constants.MACHINE_M20
params = {
    "asic_energy_consumption" : machine_params[ky]["energy_consumption"],
    "asic_hash_rate"          : machine_params[ky]["hash_rate"],
    "electricity_cost"        : 0.03,
    "machine_duration"        : machine_duration,
    "sample_rate"             : mining_constants.SAMPLE_RATE_WEEKLY,
    "mc_size"                 : 1000,
    "interest_rate"           : 0.05,
    "asic_number"             : 1,
    "asic_efficiency"         : 1.0,
    "pool_fee"                : 0.005,
    "analysis_denomination"   : mining_constants.ASSET_USD
}

sp = monte_carlo.GBMPriceLinearHashRate()
sp.fit(weekly_df, sample_rate = mining_constants.SAMPLE_RATE_WEEKLY)
m_usd_bsm = machine_npv.MachineNPV_USD_BSM(sp, params)

##########################################################################################
## 
##########################################################################################
imp.reload(implied_payback_time)
ipt    = implied_payback_time.ImpliedPayoffTime(params, m_usd_bsm, machine_df.tail(4).head(1))
ipt_df = ipt.compute(machine_type=mining_constants.MACHINE_M20, alpha=0.75, beta=0.5, MAX_DURATION=5)
start_time, hash_rate, btc_close, option_bundle_init, machine_cost, ipb_ttm  = ipt.extract_bundle(k=0)

##########################################################################################
## 
##########################################################################################
imp.reload(machine_replication)
replication_params = {
    "interest_rate" : params["interest_rate"], 
    "slippage" : 0.0
}

# Simulate RND GBM Prices.
sp._gbm_params["mu"] = 0# self._sp._gbm_params["mu"] = np.power(1 + params["interest_rate"], 1/params["sample_rate"]) - 1

ttm_vec, ttm_dt_vec, St, HRt, wt, Ht = sp.simulate(
    start_time    = start_time,
    spot_init     = btc_close,
    hashrate_init = hash_rate,
    sample_rate   = mining_constants.SAMPLE_RATE_HOURLY,
    N = 100,
    T = 2
)

res_lst = []
for lambda_C in [0.8, 0.9, 1.0, 1.1, 1.2]:
    for k in range(St.shape[1]):
        logging.info("""{} {}""".format(k, lambda_C))
        
        price_df = pd.DataFrame({"ttm" : ttm_vec, "time": ttm_dt_vec, "price": St[:, k], "wt": wt}).set_index("time")
        hedge_iv = sp._gbm_params.get("sigma") * np.sqrt(mining_constants.WEEK_IN_YEAR)

        ## Initialize dataframe for all tracked metrics 
        mr = machine_replication.MachineReplication(option_bundle_init, price_df, replication_params)
        r  = mr.process()

        tmo = machine_replication.TrueMiningOutput(mr, sp, params, option_bundle_init)
        tmo._generate_true_performance(start_time, hash_rate, lambda_C)
        result_df = tmo._generate_result_df()
        metrics = tmo._generate_metrics(machine_cost)
        logging.info(metrics)
        metrics["k"] = k
        metrics["lambda_C"] = lambda_C
        res_lst.append(metrics)


df = pd.DataFrame(res_lst)
df.to_csv("../../data/lambda_sensitivity_results_3.csv")