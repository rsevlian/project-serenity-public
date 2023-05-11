



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

import numpy as np
import monte_carlo
import data_utils
import mining_constants
import machine_npv
import logging


def machine_ccr_default_calc(params_w_loss, params_no_loss, start_time, btc_close, hash_rate, ccr):

    """
        Machine Backed USD Loan vs CCR.
    """
    sp = monte_carlo.GBMPriceLinearHashRate()
    sp.fit(weekly_df, sample_rate=mining_constants.SAMPLE_RATE_WEEKLY)

    if params_no_loss.get("BTC_DENOMINATED") == False:
        m = machine_npv.MachineNPV_USD_BSM(sp, params_no_loss)
        m_npv   = m.price(start_time, btc_close, hash_rate)  
        loan_value = m_npv * (1 / ccr)  
    else:
        m = machine_npv.MachineNPV_USD_BSM(sp, params_no_loss)
        m_npv   = m.price(start_time, btc_close, hash_rate) / btc_close
        loan_value = m_npv * (1 / ccr)

    loging.info("{} {} {}".format())

    m1   = machine_npv.MCDefaultRate(sp, params_w_loss)
    pr_default_w_loss = m1.default_rate_usd(
                                start_time, 
                                btc_close, 
                                hash_rate, 
                                mu_annual  = params_no_loss["interest_rate"], 
                                loan_value = m_npv * (1 / ccr)
                            )

    m2  = machine_npv.MCDefaultRate(sp, params_no_loss)
    pr_default_no_loss = m2.default_rate_usd(
                                start_time, 
                                btc_close, 
                                hash_rate, 
                                mu_annual  = params_no_loss["interest_rate"], 
                                loan_value = m_npv * (1 / ccr)
                            )    

    return pr_default_no_loss, pr_default_w_loss


def usd_test(start_time, btc_close, hash_rate, N, file_name):

    ## -------------------------------------------------------
    machine_params   = mining_constants.machine_params()
    ky = mining_constants.MACHINE_M20

    ## No Loss USD analysis.
    no_loss_usd_params = {
        "asic_energy_consumption" : machine_params[ky]["energy_consumption"],
        "asic_hash_rate"          : machine_params[ky]["hash_rate"],
        "electricity_cost"        : 0.00001,
        "machine_duration"        : 1.0,
        "sample_rate"             : mining_constants.SAMPLE_RATE_WEEKLY,
        "mc_size"                 : 500,
        "interest_rate"           : 0.05,
        "asic_number"             : 1,
        "asic_efficiency"         : 1.0,
        "pool_fee"                : 0.005,
        "BTC_DENOMINATED"         : False
    }

    loss_usd_params = {
        "asic_energy_consumption" : machine_params[ky]["energy_consumption"],
        "asic_hash_rate"          : machine_params[ky]["hash_rate"],
        "electricity_cost"        : 0.05,
        "machine_duration"        : 1.0,
        "sample_rate"             : mining_constants.SAMPLE_RATE_WEEKLY,
        "mc_size"                 : 500,
        "interest_rate"           : 0.05,
        "asic_number"             : 1,
        "asic_efficiency"         : 1.0,
        "pool_fee"                : 0.005,
        "BTC_DENOMINATED"         : False
    }

    ## BSM Closed Form.

    ## 
    result_lst = []
    for ccr in np.linspace(0.1, 4.0, N):
    
        # print(ccr)
        pr_default_no_loss, pr_default_loss = machine_ccr_default_calc(no_loss_usd_params, loss_usd_params, start_time, btc_close, hash_rate, ccr)

        tmp = {
            "ccr" : ccr,
            "no_loss_default" : pr_default_no_loss,
            "loss_default" : pr_default_loss
        }
        
        logging.info(tmp)
        result_lst.append(tmp)

        ## save intermediate data.
        df = pd.DataFrame(result_lst)
        df.to_csv(file_name)


def btc_test(start_time, btc_close, hash_rate, N, file_name):

    ## -------------------------------------------------------
    machine_params   = mining_constants.machine_params()
    ky = mining_constants.MACHINE_M20

    ## No Loss USD analysis.
    no_loss_btc_params = {
        "asic_energy_consumption" : machine_params[ky]["energy_consumption"],
        "asic_hash_rate"          : machine_params[ky]["hash_rate"],
        "electricity_cost"        : 0.00001,
        "machine_duration"        : 1.0,
        "sample_rate"             : mining_constants.SAMPLE_RATE_WEEKLY,
        "mc_size"                 : 500,
        "interest_rate"           : 0.05,
        "asic_number"             : 1,
        "asic_efficiency"         : 1.0,
        "pool_fee"                : 0.005,
        "BTC_DENOMINATED"         : True
    }

    loss_btc_params = {
        "asic_energy_consumption" : machine_params[ky]["energy_consumption"],
        "asic_hash_rate"          : machine_params[ky]["hash_rate"],
        "electricity_cost"        : 0.05,
        "machine_duration"        : 1.0,
        "sample_rate"             : mining_constants.SAMPLE_RATE_WEEKLY,
        "mc_size"                 : 500,
        "interest_rate"           : 0.05,
        "asic_number"             : 1,
        "asic_efficiency"         : 1.0,
        "pool_fee"                : 0.005,
        "BTC_DENOMINATED"         : True
    }

    ## BSM Closed Form.

    ## 
    result_lst = []
    ccr = 1.1
    for ccr in np.linspace(0.1, 4.0, N):
            
        pr_default_no_loss, pr_default_loss  = machine_ccr_default_calc(no_loss_btc_params, loss_btc_params, start_time, btc_close, hash_rate, ccr)
        
        tmp = {
            "ccr" : ccr,
            "no_loss_default" : pr_default_no_loss,
            "loss_default" : pr_default_loss
        }

        logging.info(tmp)
        result_lst.append(tmp)
        
        df = pd.DataFrame(result_lst)
        df.to_csv(file_name)


if __name__ == '__main__':

    ## Get historical Data.
    du = data_utils.DataUtility("cc_key")
    du._raw_network_price_df = pd.read_csv("../../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")
    du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
    du._df_feature() 
    du._aggregate_weekly_df() 

    daily_df  = du._daily_network_price_df
    weekly_df = du._weekly_network_price_df

    ## 
    k = 130

    start_time = weekly_df.reset_index().loc[k, "time"]
    hash_rate  = weekly_df.reset_index().loc[k, "hashrate"]
    hash_index = weekly_df.reset_index().loc[k, "hash_index"]
    btc_close  = weekly_df.reset_index().loc[k, "close"]

    ## -------------------------------------------------------
    btc_test(start_time, btc_close, hash_rate, 15, "../../data/btc_ccr_vs_default_130.csv")

    usd_test(start_time, btc_close, hash_rate, 15, "../../data/usd_ccr_vs_default_130.csv")

