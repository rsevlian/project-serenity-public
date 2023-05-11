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
import os, sys


home = os.environ["HOME"]
sys.path.append(f"{home}/Desktop/project-ursa/lib")

import imp
import monte_carlo
import data_utils
import mining_constants
import machine_npv
import logging


if __name__ == '__main__':

    du = data_utils.DataUtility("cc_key")

    ##
    du._raw_network_price_df = pd.read_csv("../../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")
    du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
    du._df_feature() 
    du._aggregate_weekly_df() 

    daily_df  = du._daily_network_price_df
    weekly_df = du._weekly_network_price_df

    ## 
    sp = monte_carlo.GBMPriceLinearHashRate()
    sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

    ## 
    # k = len(weekly_df) - 1
    ## 
    k = 170

    start_time = weekly_df.reset_index().loc[k, "time"]
    hash_rate  = weekly_df.reset_index().loc[k, "hashrate"]
    hash_index = weekly_df.reset_index().loc[k, "hash_index"]
    btc_close  = weekly_df.reset_index().loc[k, "close"]

    logging.info({"week": start_time, "hashrate": hash_rate, "hash-index": hash_index, "btc-close": btc_close})


    machine_params = mining_constants.machine_params()
    md_lst  = np.linspace(0.1, 7, 30)

    res_lst = []
    for md in md_lst:

        ky = "M20"
        params = {
            "asic_energy_consumption" : machine_params[ky]["energy_consumption"],
            "asic_hash_rate"          : machine_params[ky]["hash_rate"],
            "electricity_cost"        : 0.05,
            "machine_duration"        : md,
            "sample_rate"             : mining_constants.SAMPLE_RATE_WEEKLY,
            "mc_size"                 : 1000,
            "interest_rate"           : 0.05,
            "asic_number"             : 1,
            "asic_efficiency"         : 1.0,
            "pool_fee"                : 0.005,
            "analysis_denomination"   : mining_constants.ASSET_USD
        }

        logging.info("Running MC Model")

        ## Deterministic
        sp = monte_carlo.GBMPriceLinearHashRate()
        sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

        sp._gbm_params["sigma"] = 0

        m_usd_detrm = machine_npv.MachineNPV_USD_BSM(sp, params)
        usd_detrm   = m_usd_detrm.price(start_time, btc_close, hash_rate)

        # BSM Closed Form.
        sp = monte_carlo.GBMPriceLinearHashRate()
        sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

        m_usd_bsm = machine_npv.MachineNPV_USD_BSM(sp, params)
        usd_bsm   = m_usd_bsm.price(start_time, btc_close, hash_rate)

        ## BSM Binomial Lattice.
        sp = monte_carlo.GBMPriceLinearHashRate()
        sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

        m_usd_bin = machine_npv.MachineNPV_USD_Bin(sp, params)
        usd_bin   = m_usd_bin.price(start_time, btc_close, hash_rate, 100)

        # MC RND = BSM Model
        sp = monte_carlo.GBMPriceLinearHashRate()
        sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

        m_usd_mc_rnd = machine_npv.MachineNPV_MCSim(sp, params)
        usd_mc_rnd   = m_usd_mc_rnd.price(start_time, btc_close, hash_rate, mu_annual=params["interest_rate"])

        ## MC mu = zero
        sp = monte_carlo.GBMPriceLinearHashRate()
        sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

        m_usd_mc_mu_zero = machine_npv.MachineNPV_MCSim(sp, params)
        usd_mc_mu_zero = m_usd_mc_mu_zero.price(start_time, btc_close, hash_rate, mu_annual=0)
        # usd_mc_mu_zero={"mean" : 0}

        ## MC mu = empirical mean
        sp = monte_carlo.GBMPriceLinearHashRate()
        sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

        m_usd_mc_mu_emp = machine_npv.MachineNPV_MCSim(sp, params)
        usd_mc_mu_emp = m_usd_mc_mu_emp.price(start_time, btc_close, hash_rate, mu_annual=sp._gbm_params["mu"])
        # usd_mc_mu_emp={"mean" : 0}

        tmp = {
            "machine_duration" : md,
            "deterministic"    : usd_detrm,
            "bsm"              : usd_bsm,
            "binomial"         : usd_bin,
            "mc_rnd"           : usd_mc_rnd["mean"],
            "mc_mu_zero"       : usd_mc_mu_zero["mean"],
            "mc_mu_emp"        : usd_mc_mu_emp["mean"],    
        }
        logging.info(tmp)
        res_lst.append(tmp)

        df = pd.DataFrame(res_lst)

        logging.info("Saving File: ../../data/machine_price_M2_v_duration_1.csv")
        df.to_csv("../../data/machine_price_M2_v_duration_1.csv")
