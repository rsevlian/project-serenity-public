import pandas as pd 
import logging 
import requests 
import datetime 
import numpy as np 
    
import mining_constants   
  
    
def machine_market_prices(file_loc: str) -> pd.DataFrame: 
    
    """
        Return machine pricing data.
        Input: 
            file_loc - file location.
        
        Output:
            machine_prices - dataframe of prices.
    """
    machine_prices = pd.read_csv(file_loc) 
    machine_prices.time = pd.to_datetime(machine_prices.time) 
    machine_prices = machine_prices.set_index("time") 
  
    return machine_prices 

  
def daily_btc_reward(start_dt: datetime.datetime, delta_t: float) -> float: 

    """
        Generate Daily BTC Reward based.
        Input:
            start_dt - datetime for start time.
            delta_t  - delta-time: 0 - 1 in terms of years. (ex. 1=365 days)
        Output:
            quantity of BTC generated on day.  
    """

    # logging.info("start-dt {} delta_t {}".format(start_dt, delta_t)) 
    ct = start_dt + pd.Timedelta(days=int(delta_t * mining_constants.SAMPLE_RATE_DAILY)) 
      
    if (ct <= mining_constants.HALVING_0): 
        return mining_constants.HALVING_BTC_0 
      
    elif (mining_constants.HALVING_0 < ct) & (ct <= mining_constants.HALVING_1): 
        return mining_constants.HALVING_BTC_1 
      
    elif (mining_constants.HALVING_1 < ct) & (ct <= mining_constants.HALVING_2): 
        return mining_constants.HALVING_BTC_2 
      
    elif (mining_constants.HALVING_2 < ct) & (ct <= mining_constants.HALVING_3): 
        return mining_constants.HALVING_BTC_3 
      
    elif (mining_constants.HALVING_3 < ct) & (ct <= mining_constants.HALVING_4): 
        return mining_constants.HALVING_BTC_4 
      
    elif (mining_constants.HALVING_4 < ct) & (ct <= mining_constants.HALVING_5): 
        return mining_constants.HALVING_BTC_5 
  
    elif (mining_constants.HALVING_5 < ct) & (ct <= mining_constants.HALVING_6): 
        return mining_constants.HALVING_BTC_6 
  
    elif (mining_constants.HALVING_6 < ct) & (ct <= mining_constants.HALVING_7): 
        return mining_constants.HALVING_BTC_7 
  
    elif (mining_constants.HALVING_7 < ct): 
        return mining_constants.HALVING_BTC_5 
  
    
class DataUtility:

    def __init__(self, cc_key: str):
        
        self._cc_key = cc_key
        self._raw_network_price_df = None
        self._daily_network_price_df = None
        self._weekly_network_price_df = None    
        self._weekly_machine_price_df = None

    def _get_raw_df(self): 
        """
            Input: none, uses _cc_key private member.
            Oputout:
                df - merged BTC close, network-hashrate, difficulty, ... 
        """      

        ## Pull network data: time, difficulty, network hashrate.
        api_str = """https://min-api.cryptocompare.com/data/blockchain/histo/day?fsym=BTC&api_key={ck}&limit=2000""".format(ck = self._cc_key) 

        logging.info(api_str) 
        network_df = pd.DataFrame(
            requests.get(api_str).json().get("Data").get("Data")
        ) 
        network_df.time = pd.to_datetime(network_df.time, unit='s') 
        network_df = network_df[["time", "hashrate", "difficulty"]].set_index("time")

        ## Pull price data: time, close
        api_str = """https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&api_key={ck}&limit=2000""".format(ck = self._cc_key) 
        logging.info(api_str) 
        price_df = pd.DataFrame(
            requests.get(api_str).json().get("Data").get("Data")
        )
        price_df.time = pd.to_datetime(price_df.time, unit='s') 
        price_df = price_df[["time", "close"]].set_index("time") 


        self._raw_network_price_df = price_df.join(network_df).reset_index() 

    def _df_feature(self):
      
        # add some variables 
        daily_df = self._raw_network_price_df.copy()

        # daily_df["week"] = daily_df.time.dt.week 
        daily_df["month"] = daily_df.time.dt.month 
        daily_df["year"] = daily_df.time.dt.year 
      
        # Starting Date of each week. 
        daily_df["week_date"] = daily_df['time'] - pd.to_timedelta((daily_df['time'].dt.weekday - 4) % - 7, unit='d') 
      
        # reformatting 
        daily_df["time"] = daily_df.time.apply(lambda x: x.date()) 
        daily_df["week_date"] = daily_df.week_date.apply(lambda x: x.date()) 
      
        ## 
        daily_df["reward"] = daily_df.apply(lambda x: daily_btc_reward(x.time, 0), axis=1) 
         
        daily_df["hashrate"][daily_df["hashrate"] < 1] = np.NaN 
        daily_df = daily_df.ffill(axis=0) 
      
        self._daily_network_price_df =  daily_df.set_index("time") 
  
    def _aggregate_weekly_df(self): 
        
        # group by week: 
        # - add hashrate, reward since rewards additive. 
        # - avg price 
        weekly_df = self._daily_network_price_df.reset_index().groupby("week_date").agg({ 
                                        "hashrate" : "mean",   #### 
                                        "close"    : "mean", 
                                        "reward"   : "sum", 
                                        "time"     : "count" 
                                    }).copy()

        # rename time -> day_count 
        weekly_df = weekly_df.rename({"time" : "day_count"}, axis='columns') 

        # remove any row without 7 days of data. 
        weekly_df = weekly_df[weekly_df.day_count == mining_constants.DAYS_IN_WEEK] 

        # rename index. 
        weekly_df.index.names = ['time'] 

        # daily reward shoud be adjusted       
        weekly_df['hash_index'] = (weekly_df.close * weekly_df.reward) / (weekly_df.hashrate * mining_constants.DAYS_IN_WEEK) 
        weekly_df['hash_index_return'] = (weekly_df.hash_index - weekly_df.hash_index.shift(1)) / weekly_df.hash_index.shift(1)
        weekly_df['hash_index_return_log'] = np.log(weekly_df.hash_index / weekly_df.hash_index.shift(1)) 
        weekly_df['hash_index_diff'] = weekly_df.hash_index - weekly_df.hash_index.shift(1) 

        weekly_df["price_returns_log"] = np.log(weekly_df.close / weekly_df.close.shift(1)) 
        weekly_df["price_returns"] = (weekly_df.close - weekly_df.close.shift(1)) / weekly_df.close.shift(1) 

        weekly_df["hashrate_returns_log"] = np.log(weekly_df.hashrate / weekly_df.hashrate.shift(1)) 
        weekly_df["hashrate_returns"] = weekly_df.hashrate / weekly_df.hashrate.shift(1) - 1 

        self._weekly_network_price_df =  weekly_df 
  
    def generate_weekly_df(self):

        ## pull and join raw data.
        self._get_raw_df() 

        ## generate features.
        self._df_feature() 

        ## 
        self._aggregate_weekly_df() 
    
    def _join_machine_price(self, machine_prices: pd.DataFrame, HR_ROLLING_AVG: int):

        start_time = machine_prices.index[0] 
        end_time = machine_prices.index[-1] 
        
        df = self._daily_network_price_df.copy()
        df["hashrate"] = df["hashrate"].rolling(HR_ROLLING_AVG).mean() 
        
        df = df.join(machine_prices, how="left") 
        df = df[[ 
                    "close",  
                    "hashrate",  
                    mining_constants.MACHINE_S9,  
                    mining_constants.MACHINE_S17,  
                    mining_constants.MACHINE_S19J,   
                    mining_constants.MACHINE_M20,  
                    mining_constants.MACHINE_M30 
                ]] 
        
        df = df.fillna(method="ffill") 
        df = df[ (df.index > start_time) & (df.index < end_time) ] 
        
        SAMPLE_RATE = 7 ## need to sample every 7 days for weekly.
        df = df.iloc[::SAMPLE_RATE, :] 
        
        return df

    def generate_weekly_machine_price_df(self, machine_file_loc : str, HR_ROLLING_AVG: int): 
    
        
        ## pull and join raw data.
        self._get_raw_df()

        ## generate features.
        self._df_feature() 

        ## 
        self._aggregate_weekly_df() 

        ## Pull machine pricing file.
        machine_prices = machine_market_prices(machine_file_loc)
        

        self._weekly_machine_price_df = self._join_machine_price(machine_prices, HR_ROLLING_AVG)
        