import sys, os 
  
import pandas as pd 
import numpy as np 
import math 

import mining_constants  
import bsm_utils 
import multiprocessing 
from multiprocessing import Pool 

import matplotlib.pyplot as plt 
import logging 
from enum import Enum 
  
  
############################################################################### 
## Continous Delta Hedging 
############################################################################### 

def check_input(params):
    
    tok_list = [
            "sigma", "K", "T", "Q", "interest_rate", 
            "sample_rate", "contract_type", "contract_direction"
        ]
    
    for tok in tok_list:
        if tok not in params:
            raise Exception("Missing Argument {} in {}".format(tok, params))

def test_valid_DH(df, itm, params):

    if itm:
        # print("ITM ", df.purchase_quantity.sum(), params.get("Q"), round(df.purchase_quantity.sum() - params.get("Q"), 3))
        assert round(df.purchase_quantity.sum() - params.get("Q"), 3) == 0

    if not itm:
        # print("OTM ", df.purchase_quantity.sum(), params.get("Q"))
        assert round(df.purchase_quantity.sum(), 3) == 0.0

def state_machine(t, price_t, delta_t, slippage_t, params): 

    """ 
        Simulate continues delta hedging dataframe.
        Input:
            price_t    - [Tx1] price vector.
            sigma_iv_t - [Tx1] sigma_iv vector.
            delta_t    - [Tx1] delta vector
            slippage_t - [Tx1] slippage vector
            params:
                - interest rate
                - asset_per_contract
                - sample_rate 
    """ 

    ## 
    interest_rate        = params.get("interest_rate")
    asset_per_contract   = params.get("Q")
    sample_rate          = params.get("sample_rate")

    ## initialized dataframe 
    df = pd.DataFrame({ 
            "t"        : t,  
            "delta"    : delta_t, 
            "price"    : price_t, 
            "slippage" : slippage_t 
        }) 

    ## Compute the TTM for each row. 
    T_end = len(df) 

    ## Compute the per row discount factor. 
    # df["forward_discount"] = np.exp(interest_rate * (params['ttm'] - t)) 
    # 10Jan23 SJL - Introducing below change so all pricing is present value  
    df["discount_factor"] = np.exp(-interest_rate * t) 

    ## Cmpute the Delta-Delta (change in delta needed to trigger purchases.) 
    df["delta_delta"] = df["delta"].diff() 
    df["delta_delta"] = df["delta_delta"].fillna(delta_t[0]) 

    # e.g. if delta_delta > 0, we buy shares @p, but with slippage, we buy shares 
    # at p(1+eps). 
    # Case:        Qty      Purchase Value 
    # No slippage:   q      -p 
    # With slippage: q      -p(1+ep)   
    df["slippage_multiplier"] = (1 + df["slippage"] * np.sign(df["delta_delta"])) 

    ## Purchase Quantity: How many shares required to purchase to hedge 
    df["purchase_quantity"]   = asset_per_contract * df["delta_delta"] 

    ## Value in USD to purchase shares including slippage 
    # Purchase Quantity > 0 <==> Purchase Value < 0, unless you are being paid to buy shares :) 
    df["purchase_value"]      = df["purchase_quantity"] * df["price"] * df["slippage_multiplier"] 

    ## Discounted value into future when option expires  
    df["discounted_purchase_value"] = df["discount_factor"] * df["purchase_value"] 

    

    return df

def _contract_itm(S, K, contract_type):
    
    ## Compute ITM or OTM. 
    if ( contract_type == mining_constants.CONTRACT_CALL_OPTION ):         
        return S > K
    
    elif ( contract_type == mining_constants.CONTRACT_PUT_OPTION ): 
        return S < K
    
    else: 
        raise Exception("Invalid Contract Type") 

# def hedging_cost(purchase_cost, params):
def _hedging_cost_calc(purchase_cost, itm, df, params):
    """
        Computes total hedging cost given contract type/parameters & expiry conditions, 
        
        Input:
            purchase_cost - total delta hedging cost.
            itm           - boolean condition if contract expired ITM/OTM
            df            - discount factor at expiry 
            params - [dict]
                - contract_direction : LONG | SHORT
                - contract_type      : CALL | PUT
                - K                  : strike
        
        Output:
            hedging_cost 

    """

    dir_long  = params.get("contract_direction") == mining_constants.DIRECTION_LONG_CONTRACT
    type_call = params.get("contract_type") == mining_constants.CONTRACT_CALL_OPTION
    K = params.get("K")
    Q = params.get("Q")

    if dir_long and itm and type_call:
        hedging_cost = -1 * purchase_cost + K * Q * df 
        # print(purchase_cost,  K * Q * df, hedging_cost)
    
    # elif dir_long and itm and (not type_call):
    #     hedging_cost = purchase_cost + K * Q * df

    elif dir_long and (not itm) and type_call:
        # print(purchase_cost,  hedging_cost, )
        hedging_cost = purchase_cost

    # elif dir_long and (not itm) and (not type_call):
    #     hedging_cost = purchase_cost
    # 
    # elif (not dir_long) and itm and type_call:
    #     hedging_cost = purchase_cost + K * Q * df 
    # 
    # elif (not dir_long) and itm and (not type_call):
    #     hedging_cost = purchase_cost - K * Q * df
    # 
    # elif (not dir_long) and (not itm) and type_call:
    #     hedging_cost = purchase_cost
    # 
    # elif (not dir_long) and (not itm) and (not type_call):
    #     hedging_cost = purchase_cost

    else:
        raise Exception("Hedging Metrics not supported {} {} {}".format(dir_long, itm, type_call))

    return hedging_cost

def delta_hedging_metrics(df, params):

    """ 
        Computes window-level metrics, taking into account L/S direction. 
        Input:
            df           - 
            params       - 
                quote_iv - 
                strike   -
        Output:
    """  

    ## 
    price_t = df.price.values
    itm = _contract_itm(
        S = price_t[-1], 
        K = params.get("K"), 
        contract_type = params.get("contract_type")
    )

    ## Total "hedging cost" of just cash component (no shares) 
    purchase_cost         = df["discounted_purchase_value"].sum()  ## w/ slippage
    discount_factor       = df["discount_factor"].iloc[-1] 

    
    hedging_cost = _hedging_cost_calc(purchase_cost, itm, discount_factor, params)
    
    bsm_price = bsm_utils.bsm_price(
                            sigma=  params.get("sigma"), 
                            S = price_t[0],
                            K = params.get("K"), 
                            r = params.get("interest_rate"), 
                            T = params.get("T"),
                            contract_type = params.get("contract_type")
                        )

    agg_bsm = params.get("Q") * bsm_price * discount_factor
    
    test_valid_DH(df, itm, params)

    return { 
        "ITM"                       : itm, 
        "Final Spot"                : price_t[-1],
        "K"                         : params.get("K"), 
        "purchase_cost"             : purchase_cost, 
        "hedging_cost"              : hedging_cost,
        "bsm_price"                 : bsm_price , 
        "agg_bsm_price"             : agg_bsm, 
        "hedging_err"               : (agg_bsm + hedging_cost)/agg_bsm,
        "excersized_cash"           : (1.0 * itm) * params.get("K") * params.get("Q"),
        "discount_factor"           : discount_factor,
        ## save df.
        "df"                        : df     
    } 

def simulate_delta_hedging(t, quote_iv, price_t, delta_t, slippage_t, params): 

    check_input(params)
    df = state_machine(t, price_t, delta_t, slippage_t, params) 
    metrics = delta_hedging_metrics(df, params) 

    return metrics

############################################################################### 
## Multiprocessing Run Simulator. 
############################################################################### 
def run_delta_hedge(simulator, t, quote_iv, PRICE_t, DELTA_t, SLIPPAGE_t, params, n_process=8): 

    """ 
    Run Delta Hedge Simulator.
    Input:
        simulator  : {simulate_continuous_delta_hedging | simulate_threshold_delta_hedging (optional)} 
        t          : T x 1 vector of time to maturity
        quote_iv   : vector of length N containing quote IV for each window 
        PRICE_t    : matrix of T x N price vectors (historical & monte-carlo models) 
        SIGMA_IV_t : matrix of T x N implied vols as measured from the market 
        DELTA_t    : matrix of T x N delta hedging values (historical & monte-carlo models) 
        SLIPPAGE_t : matrix of T x N slippage values (historical & monte-carlo models) 
        params     : additional params dictionary for simulator. 

    """ 

    # logging.info( 
    #     "simulator {} t {} quote_iv {} price {} delta {} slippage {} params {}".format(
    #         simulator, t, quote_iv[0:1], np.shape(PRICE_t), len(DELTA_t), np.shape(SLIPPAGE_t), params
    #     ) 
    # )

    with multiprocessing.Pool(n_process) as pool: 

        ## get size of montecarlos models 
        mc_size = np.shape(PRICE_t)[1] 

        ## allocate matrix to individual monte-carlo tests. 
        args = [(t, quote_iv[k], PRICE_t[:, k], DELTA_t[:, k], SLIPPAGE_t[:, k], params) for k in range(mc_size)] 


        ## star-map 
        results = pool.starmap(simulator, args) 

    return pd.DataFrame(results)


