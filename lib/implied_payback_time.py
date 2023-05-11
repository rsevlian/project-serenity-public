import pandas as pd 
  
import monte_carlo 
# import dcf_utils 
# import npv_dcf_sim 
# import npv_bsm_sim
import datetime 
import numpy as np 
import logging 
  
from machine_npv import MachineNPV

# import os, sys 
# 
# home = os.environ["HOME"] 
# sys.path.append(f"{home}/go/src/github.com/anchorlabsinc/anchorage/source/python/quant_lib/anchoragequantlib/options_lib") 
# import bsm_utils   



def compute_implied_option_bundle(option_df, mprice):  

    # stochastic_process 
    # machine_pricing_params  
    # MAX_DURATION = 15 
    
    try:  
        idx_ipt = np.min(np.where(option_df.reward_value.cumsum() >= mprice)) 
          
        npv = option_df.iloc[0:idx_ipt + 1].reward_value.sum() 
        ip_ttm = option_df.iloc[idx_ipt].ttm 
          
        assert npv >= mprice 

        return idx_ipt, ip_ttm, npv 

    except:
          
        return np.NaN, np.NaN, np.NaN 
 

def compute_effective_value(option_df, alpha=0.75):
      
    # return option_df 
    lifetime_value = option_df.reward_value.sum() 

    try:  
        idx_upper = np.max(np.where(option_df.reward_value.cumsum() <= lifetime_value * alpha)) 
        idx_lower = np.min(np.where(option_df.reward_value.cumsum() > lifetime_value * alpha)) 
      
        eff_npv = option_df.iloc[0:idx_upper].reward_value.sum() 
        eff_ttm = option_df.iloc[idx_upper].ttm 
      
        return idx_upper, eff_ttm, eff_npv 
          
    except:
        
        return np.NaN, np.NaN, None 

  
def compute_feasible_value(option_df, beta=0.25):  
      
  
    try:  
        idx_upper = np.max(np.where(option_df.delta >= beta)) 
        idx_lower = np.min(np.where(option_df.delta <  beta))

        fsbl_npv = option_df.iloc[0:idx_upper].reward_value.sum() 
        fsbl_ttm = option_df.iloc[idx_upper].ttm 

        return idx_upper, fsbl_ttm, fsbl_npv 

    except: 

        return np.NaN, np.NaN, None 


class ImpliedPayoffTime():
  
    def __init__(self, params: dict, machine_npv: MachineNPV, df_merge: pd.DataFrame): 
        
        ## 
        self._params = params

        ## needed stochastic process. 
        self._machine_npv = machine_npv 
          
        ## needed data. 
        self._df_merge = df_merge 
  
        ## 
        self._option_df_dict = {} 

        self._ipt_df = None
  
    def _precompute_max_option_bundles(self, machine_type, MAX_DURATION=5): 

        for k in range(len(self._df_merge)): 

            ## 
            start_time = self._df_merge.index[k].to_pydatetime().date()
            hash_rate  = self._df_merge.iloc[k]["hashrate"] 
            btc_close  = self._df_merge.iloc[k]["close"] 

            # logging.info("precomputing option_df: k {} time {} hash-rate {} btc {}".format(k, start_time, hash_rate, btc_close))         

            tmp_dict = self._params
            tmp_dict["machine_duration"] = MAX_DURATION 

            ##
            self._machine_npv.price(start_time, btc_close, hash_rate)
            self._option_df_dict[start_time] = self._machine_npv._option_df
            
    def _compute_machine_values(self, machine_type, alpha, beta): 

        res_lst = [] 
        for k in range(len(self._df_merge)): 

            time_start = self._df_merge.index[k].to_pydatetime().date()
            hash_rate  = self._df_merge.iloc[k]["hashrate"] 
            btc_close  = self._df_merge.iloc[k]["close"] 
            mprice     = self._df_merge.iloc[k][machine_type] 

            # precomputed option bundle df 
            option_df  = self._option_df_dict[time_start] 

            # machine payback time. 
            ipb_idx, ipb_ttm, ipb_npv = compute_implied_option_bundle(option_df, mprice) 

            # compute effective value 
            eff_idx, eff_ttm, eff_npv = compute_effective_value(option_df, alpha) 

            # compute effective value 
            fsbl_idx, fsbl_ttm, fsbl_npv = compute_feasible_value(option_df, beta) 

            tmp = { 
                "time" : time_start, 
                "hash_rate" : hash_rate, 
                "btc_close" : btc_close, 
                "machine_price" : mprice, 

                #  
                "ipb_npv" : ipb_npv, 
                "ipb_ttm" : ipb_ttm, 
                "ipb_idx" : ipb_idx, 

                #  
                "eff_npv" : eff_npv, 
                "eff_ttm" : eff_ttm, 
                "eff_idx" : eff_idx, 

                #  
                "fsbl_npv" : fsbl_npv, 
                "fsbl_ttm" : fsbl_ttm, 
                "fsbl_idx" : fsbl_idx 
            } 

            # logging.info(tmp) 

            res_lst.append(tmp) 

        self._ipt_df = pd.DataFrame(res_lst)

    def compute(self, machine_type, alpha, beta, MAX_DURATION=5):

        self._precompute_max_option_bundles(machine_type, MAX_DURATION)
        
        self._compute_machine_values(machine_type, alpha, beta)

        return self._ipt_df

    def extract_bundle(self, k): 
        """ 
             extract_bundle: 
              -  
        """ 
        machine_price = self._ipt_df.iloc[k].machine_price
        start_time    = self._df_merge.index[k].to_pydatetime().date()
        hash_rate     = self._df_merge.iloc[k]["hashrate"] 
        btc_close     = self._df_merge.iloc[k]["close"] 

        ipb_idx       = int(self._ipt_df.iloc[k].ipb_idx) 
        ipb_ttm       = self._ipt_df.iloc[k].ipb_ttm
        option_bundle = self._option_df_dict[start_time].iloc[0:ipb_idx] 
     
        return start_time, hash_rate, btc_close, option_bundle, machine_price, ipb_ttm
      




