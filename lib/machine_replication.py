import os 
import sys 
import pandas as pd 
import numpy as np 
import mining_constants 
import logging 

import bsm_utils 

# import hedging_constants 
import delta_hedge 

import backtest 
import crypto_compare 
import matplotlib.pyplot as plt  

from monte_carlo import GBMPriceLinearHashRate

import data_utils

def hourly_btc_price(start_dt, end_dt, cc_key): 

    cc = crypto_compare.cc_utility(cc_key) 

    btc_hourly = cc.cc_hourly("BTC", "USD", start_dt, end_dt) 
    btc_price = btc_hourly[["time", "open"]].set_index("time")  #.resample("1D").first() 

    return btc_price 

      
class MachineReplication(): 
  
    def __init__(self, option_bundle, price_df, params): 

        """ 
            option_bundle: bundle we need to hedge  
            ttm / ttm_dt  s
            reward 
            sigma   
            strike 

            price_df:  
            index: datetime  
            ttm : (years from current time) 
            price:  
            hourly block reward 

            _bundle_delta_hedge_df - list[pandas.dataframe]            
        """
    
        self._option_bundle = option_bundle 

        self._price_df = price_df


        self._params = params

        ## intermediate data.
        self._spot_init = None
        self._start_dt  = None
        self._bundle_delta_hedge_df = [] 
        self._agg_df = None
        self._raw_df = None


    def _setup(self):
        
        ## prefilter price_df 
        st = self._option_bundle.ttm_dt.values[0]
        et = self._option_bundle.ttm_dt.values[-1]

        self._price_df  = self._price_df.loc[st:et]
        self._spot_init = self._price_df.iloc[0].price
        self._start_dt  = self._option_bundle.ttm_dt[0]

        assert (self._price_df.reset_index().iloc[0].time == st) & (self._price_df.reset_index().iloc[-1].time == et)

    def _delta_hedge_single_call(self, start_dt, end_dt, strike, quantity, hedge_iv): 
        """
            Takes single strip mining option equivalent and performs delta-hedging operation.
            Input:
                start_dt - simulation start time
                end_dt   - expiry of single strip option.
                strike   - strike price of single strip
                quantity - reward quantity of single strip.
                hedg_iv  - sigma used to hedge strip (can be origination iv or other)
            
            Output:
                df_result:
                    - agg_delta
                    - purchase_value
                    - purchase_quantity
                    - time
        """

        
        # logging.info("st - {}, et - {}, k - {} q - {} sigma - {}".format(start_dt, end_dt, strike, quantity, hedge_iv))
        
        ## Get the  
        price_t = self._price_df.loc[start_dt:end_dt].price.values 
        ttm_vec = self._price_df.loc[start_dt:end_dt].ttm.values 
        dt_vec  = self._price_df.loc[start_dt:end_dt].index 

        ## S       - T x 1 vector of prices.
        ## ttm_vec - T x 1 vector of ttms.     
        delta_t  = bsm_utils.bsm_delta( 
                                     sigma = hedge_iv,
                                     S = price_t,
                                     K = strike,
                                     T = ttm_vec[-1] - ttm_vec,
                                     r = self._params.get("interest_rate"),
                                     contract_type=mining_constants.CONTRACT_CALL_OPTION 
                                 ) 

        ## Backtest the delta hedging operation. 
        args = { 
            "contract_type"        : mining_constants.CONTRACT_CALL_OPTION, 
            "contract_direction"   : mining_constants.DIRECTION_LONG_CONTRACT, 
            "sample_rate"          : mining_constants.SAMPLE_RATE_HOURLY, 
            "interest_rate"        : self._params.get("interest_rate"), 
            "Q"                    : quantity, 
            "T"                    : ttm_vec[-1],
            "K"                    : strike,
            "sigma"                : hedge_iv  
        } 

        ## Compute the delta hedging df.         
        # delta_hedge.check_input(args)
        metrics = delta_hedge.simulate_delta_hedging(ttm_vec, hedge_iv, price_t, delta_t, self._params.get("slippage"), args)
                
        metrics["df"]["agg_delta"]  = metrics["df"]["delta"] * quantity
        metrics["df"]["time"] = dt_vec
        metrics["end_dt"]     = end_dt 
        metrics["q_short"]    = metrics["df"].purchase_quantity.sum()

        return metrics
  
    def _replicate_option_bundle(self): 

        ## Initialize dataframe for all tracked metrics 
        self._bundle_delta_hedge_df = [] 


        ## Iterate through each options and collect 
        K_max = len(self._option_bundle)  

        # print("len -> ", len(self._option_bundle), "K-max: ", K_max)
        for k in range(K_max): 

            ## get current weeks paramters. 
            current_option = self._option_bundle.iloc[k] 
            quote_iv       = current_option.sigma 
            end_dt         = current_option.ttm_dt 
            strike         = current_option.strike 
            quantity       = current_option.reward 
            hdg_iv         = current_option.sigma 

            # logging.info( 
            #         "k {} start_dt {} end_dt {} spot {} strike {} iv {} q {}".format(k, self._start_dt, end_dt, self._spot_init, strike, hdg_iv, quantity) 
            #     ) 

            result = self._delta_hedge_single_call(self._start_dt, end_dt, strike, quantity, hdg_iv) 

            self._bundle_delta_hedge_df.append(result) 

    def _aggregate_sim_dfs(self): 

        ## Initialize dataframe for all tracked metrics 
        raw_df = self._price_df.copy().reset_index()[["time", "price"]] 

        # raw_df["agg_delta"] = 0 
        # raw_df["purchase_value"] = 0 

        for k in range(len(self._bundle_delta_hedge_df)): 

            df_curr = self._bundle_delta_hedge_df[k]["df"]
            df_curr["k"] = k 

            raw_df = pd.concat( 
                             [raw_df, df_curr[["k", "time", "price", "agg_delta", "purchase_quantity", "purchase_value"]]]  
                         ) 

        ## assign raw_df of concatentated dictionary elements.
        self._raw_df = raw_df

    def _generate_metrics(self):



       ## perform groupby operation by time index.
        agg_df = self._raw_df.groupby("time").agg({ 
                                     "agg_delta": "sum",  
                                     "purchase_quantity" : "sum", 
                                     "purchase_value" : "sum",  
                                     "price" : "first" 
                                 }) 

        agg_df["cash_balance"]    = agg_df["purchase_value"].cumsum() 
        agg_df["exercised_cash"]  = 0  ## leaving zeros here.
        agg_df["exercised"]       = 0 
        agg_df["strike"]          = 0

        for k in range(len(self._bundle_delta_hedge_df)): 

            end_dt   = self._option_bundle.iloc[k].ttm_dt 
            json_res = self._bundle_delta_hedge_df[k] 
            
            # print(json_res.get('excersized_cash'))
            
            agg_df['exercised_cash'].loc[end_dt] = json_res.get('excersized_cash')
            agg_df['exercised'].loc[end_dt]      = json_res.get('ITM')

            # print(" before - ", self._agg_df["strike"].loc[end_dt], json_res.get("strike"))
            
            agg_df["strike"].loc[end_dt]         = json_res.get("K")
            


        # ## total cash balance 
        agg_df["net_cash_balance"] = agg_df["cash_balance"] - agg_df["exercised_cash"].cumsum() 

        machine_cost    = self._option_bundle["reward_value"].sum() 
        hedging_balance = agg_df.iloc[-1].net_cash_balance 
        tracking_error  = (machine_cost - hedging_balance)
        delta_init      = agg_df.iloc[0].agg_delta 

        self._agg_df = agg_df

        results = { 
            "machine_cost" : machine_cost, 
            "hedging_balance" : hedging_balance, 
            "tracking_error" : tracking_error / machine_cost, 
            "delta_init" : delta_init            
        } 

        return results

    def process(self):
        self._setup()
        self._replicate_option_bundle()
        self._aggregate_sim_dfs()
        return self._generate_metrics()


class TrueMiningOutput():

    def __init__(self, mr: MachineReplication, sp: GBMPriceLinearHashRate, params: dict, opt_init):

        self._sp = sp
        self._params = params
        self._opt_init = opt_init
        self._mr = mr
        self._opt_tr = None
        self._result_df = None
        self._metrics = None
    
    # 
    def _generate_true_performance(self, start_time, hashrate, lambda_C=1):

        ## generate a model hashrate & reward and ttm vector. 
        sample_rate = mining_constants.SAMPLE_RATE_WEEKLY

        ttm_dt_vec = self._opt_init.ttm_dt.values
        ttm_vec    = self._opt_init.ttm.values

        HRt = self._sp._hashrate_forecast(hashrate, ttm_vec, lambda_C) 
        wt_daily = np.array([data_utils.daily_btc_reward(start_time, dt) for dt in ttm_vec]) 

        ## samples per day
        C = (mining_constants.DAYS_IN_YEAR / sample_rate)  
        
        logging.info("simulate: daily_reward x (365 / {}) = daily_reward x ({})".format(sample_rate, C)) 


        # generate the dataframe 
        option_df = pd.DataFrame(data={"ttm" : ttm_vec,  "ttm_dt" : ttm_dt_vec, "network_hashrate" : HRt, "network_reward": C*wt_daily}) 
    
                
        option_df["reward_nominal"] = option_df["network_reward"] * (self._params["asic_hash_rate"] / option_df["network_hashrate"]) 
    
    
        option_df["sigma"]  = self._sp._gbm_params["sigma"] * np.sqrt(self._params["sample_rate"]) 

        # compute weekly cost of BTC production i.e. the strike price of the option. 
        weekly_cost = self._params["electricity_cost"] * self._params["asic_energy_consumption"] * \
                            mining_constants.DAYS_IN_WEEK * mining_constants.HOURS_IN_DAY 

        option_df["weekly_cost"] = weekly_cost  
        option_df["strike"] = option_df["weekly_cost"] / option_df["reward_nominal"] 
        
        ## need to get correct timestamps.
        option_df = option_df.merge(self._mr._agg_df.reset_index()[["time", "price"]], left_on ="ttm_dt", right_on="time")

        option_df["reward"] = option_df["reward_nominal"]* (1*( option_df["price"] > option_df["strike"]))
        
        self._opt_tr = option_df.drop("price", axis='columns')


    def _generate_result_df(self):
        
        import machine_npv

        opt_init = self._opt_init[["ttm", "ttm_dt", "network_hashrate", "reward", "strike"]]
        opt_tr = self._opt_tr[["ttm", "ttm_dt", "network_hashrate", "reward", "strike"]]
        df = self._opt_init.merge(self._opt_tr, left_on="ttm_dt", right_on="ttm_dt", how="left", suffixes=("_init", "_true") )
        
        rs = self._mr._agg_df.copy().reset_index()
        rs["ttm_dt"] = rs["time"]
        rs = rs[["ttm_dt", "price", "agg_delta", "purchase_quantity", "cash_balance", "net_cash_balance"]]

        df = rs.merge(df, left_on="ttm_dt", right_on="ttm_dt", how="left")
        df["dh_btc_short_position"] = -df["purchase_quantity"].cumsum()
        df["total_reward_init"]     = df.reward_init.fillna(0).cumsum()
        df["total_reward_true"]     = df.reward_true.fillna(0).cumsum()

        df["net_btc_position_init"]      = df["total_reward_init"] + df["dh_btc_short_position"]
        df["net_btc_position_true"]      = df["total_reward_true"] + df["dh_btc_short_position"]

        self._result_df = df
        return self._result_df.set_index("ttm_dt")

    def _generate_metrics(self, machine_cost):


        x1 = (self._result_df["reward_init"] * (self._result_df["price"] - self._result_df["strike_init"])).dropna()
        weekly_liquidation_pnl_init = x1[x1>0].sum()


        x2 = (self._result_df["reward_true"] * (self._result_df["price"] - self._result_df["strike_init"])).dropna()
        weekly_liquidation_pnl_true = x2[x2>0].sum()        

        init_short_size = self._result_df.cash_balance.values[0]
        net_cash = self._result_df.net_cash_balance.values[-1]
        price_end = self._result_df.price.values[-1]
        btc_shortfall_init = self._result_df.net_btc_position_init.values[-1] 
        btc_shortfall_true = self._result_df.net_btc_position_true.values[-1] 

        error_init = (net_cash - machine_cost + btc_shortfall_init * price_end)/machine_cost
        error_true = (net_cash - machine_cost + btc_shortfall_true * price_end)/machine_cost

        error_unhedged_init = (weekly_liquidation_pnl_init - machine_cost) / machine_cost
        error_unhedged_true = (weekly_liquidation_pnl_true - machine_cost) / machine_cost

        self._metrics = { 
            "machine_cost" : machine_cost,
            "tracking_error_ideal" : error_init,
            "tracking_error_true": error_true,
            "short_shortfall_init" : btc_shortfall_init,
            "short_shortfall_true" : btc_shortfall_true, 
            "weekly_liq_unhedged_init" : weekly_liquidation_pnl_init,
            "weekly_liq_unhedged_true" : weekly_liquidation_pnl_true,
            "unhedged_tracking_error_init" : error_unhedged_init,
            "unhedged_tracking_error_true" : error_unhedged_true,
        }
        return self._metrics

    def plot_simulation(self, machine_cost):

        df = self._result_df
        qmax = df["dh_btc_short_position"].abs().max()

        plt.figure(figsize=(15, 6))

        plt.subplot(2, 3, 1)
        plt.plot(df.price, label="price") 
        plt.plot(df.strike_true.dropna(), linestyle='--', label="strike (TR)") 
        plt.plot(df.strike_init.dropna(), linestyle='--', label="strike") 
        plt.title("Price") 
        plt.legend() 
        plt.grid() 

        plt.subplot(2, 3, 2)
        plt.plot(df.agg_delta) 
        plt.title("Aggregate Delta") 
        plt.grid() 

        plt.subplot(2, 3, 3)
        plt.plot(df.total_reward_true, color="g", label="Total BTC Reward (I)")
        plt.plot(df.total_reward_init, color="r", label="Total BTC Reward (TR)")
        plt.title("Total Mining Reward")
        plt.grid() 

        plt.subplot(2, 3, 4)
        plt.plot(df.dh_btc_short_position, color="g", label="Short BTC Position")
        plt.title("Short BTC Position")
        plt.grid() 
        plt.legend()
        
        plt.subplot(2, 3, 5)
        plt.plot(df.net_btc_position_init, label="Net BTC Position (Init)")
        plt.plot(df.net_btc_position_true, label="Net BTC Position (TR)")
        plt.title("Net BTC")
        plt.grid()
        plt.legend()

        plt.subplot(2, 3, 6) 
        plt.plot(df.cash_balance, label="Long Cash Position") 
        plt.plot(df.net_cash_balance, label="Net Cash Position") 
        plt.axhline(machine_cost, label="Machine Cost(t=0)") 
        plt.ylim((0, df.cash_balance.max()*1.1))
        plt.title("Cash Position")
        plt.legend() 
        plt.grid()
        plt.show()

        logging.info(self._metrics)


