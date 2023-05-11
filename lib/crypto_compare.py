import requests 
import logging 
import pandas as pd 
from datetime import date, timedelta, datetime 
  
  
class cc_utility: 
     
    def __init__(self, ckey): 
        self._key = ckey 

    def _cc_hourly_snapshot(self, base_sym, quote_sym, end_time, nsamples=2000): 
  
        """ 
            get hourly 
        """ 
        try:

            end_time_ts = int(end_time.timestamp()) 
            
            url = """https://min-api.cryptocompare.com/data/v2/histohour?fsym={bs}&tsym={qs}&limit={ns}&api_key={ck}&toTs={et}""".format( 
                bs=base_sym, qs=quote_sym, ns=nsamples, ck=self._key, et=end_time_ts 
            ) 

            logging.info(f"url - {url}") 
            tmp = requests.get(url) 

            df = pd.DataFrame(tmp.json()["Data"]["Data"]) 
            df.time = pd.to_datetime(df.time, unit="s") 

            df.set_index("time", inplace=True) 

            df = df[["open", "high", "low", "close", "volumefrom", "volumeto"]] 
            df.reset_index(inplace=True) 

            return df 

        except Exception as e: 
            raise Exception("No CC data for symbol: ", base_sym, quote_sym, end_time) 
        
        return [] 
  
    def cc_hourly(self, base_sym, quote_sym, start_dt, end_dt): 

        et = end_dt 
        result_lst = [] 

        while True: 

            tmp_df = self._cc_hourly_snapshot(base_sym, quote_sym, et) 
            result_lst.append(tmp_df) 

            ## if we have time < start-time, then stop loop. 
            if tmp_df.iloc[0].time < start_dt: 
                break 
            else: 
                et = tmp_df.iloc[0].time 

            df = pd.concat(result_lst) 

            df = df.set_index("time") 
            df = df.sort_index() 
            df = df.drop_duplicates(keep="first") 

            ## filter start/end dates. 
            df = df[(df.index >= start_dt) & (df.index <= end_dt)] 
            df = df.reset_index() 

            ## Make sure not missing any data 
            if sum(pd.to_datetime(df["time"]).diff() > pd.Timedelta(1, "h")) > 0: 
                logging.warning("There is a missing timestamp!") 

        return df 

    def _cc_daily_snapshot(self, base_sym, quote_sym, end_time, nsamples=2000): 

        try: 
            end_time_ts = int(end_time.timestamp()) 
            url = """https://min-api.cryptocompare.com/data/v2/histoday?fsym={bs}&tsym={qs}&limit={ns}&api_key={ck}&toTs={et}""".format( 
                bs=base_sym, qs=quote_sym, ns=nsamples, ck=self._key, et=end_time_ts 
            ) 

            logging.info(f"url - {url}") 
            tmp = requests.get(url) 

            df = pd.DataFrame(tmp.json()["Data"]["Data"]) 
            df.time = pd.to_datetime(df.time, unit="s") 

            df.set_index("time", inplace=True) 

            df = df[["open", "high", "low", "close", "volumefrom", "volumeto"]] 
            df.reset_index(inplace=True) 
        
            return df 

        except Exception as e: 
            raise Exception("No CC data for symbol: ", base_sym, quote_sym, end_time) 
            
            return [] 
  
    def cc_daily(self, base_sym, quote_sym, start_dt, end_dt): 

        et = end_dt 
        result_lst = [] 

        while True: 

            ## 
            tmp_df = self._cc_daily_snapshot(base_sym, quote_sym, et) 
            result_lst.append(tmp_df) 

            ## if we have time < start-time, then stop loop. 
            if tmp_df.iloc[0].time < start_dt: 
                break 
            else: 
                et = tmp_df.iloc[0].time 

        ## 
        df = pd.concat(result_lst) 
        df = df.set_index("time") 
        df = df.sort_index() 
        df = df.drop_duplicates(keep="first") 

        ## filter start/end dates. 
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        df = df.reset_index() 

        return df 
          
    def _find_month_end(self, month, year):

        last_day = date(year + int(month / 12), (month % 12) + 1, 1) - timedelta(days=1) 
        return last_day 

    def cc_monthly(self, base_sym, quote_sym, start_dt, end_dt): 

        # Pull daily prices 
        df_daily = self.cc_daily(base_sym=base_sym, quote_sym=quote_sym, start_dt=start_dt, end_dt=end_dt) 

        # Find month end prices 
        df_daily["year"] = df_daily["time"].apply(lambda x: x.year) 
        df_daily["month"] = df_daily["time"].apply(lambda x: x.month) 
        df_daily["month_end"] = df_daily.apply(lambda x: self._find_month_end(x["month"], x["year"]), axis=1) 
        df_daily["month_end"] = pd.to_datetime(df_daily["month_end"]) 
        idx_month_end = df_daily["time"] == df_daily["month_end"] 
        df_monthly = df_daily[idx_month_end] 

        # Drop temporary columns 
        df_monthly.drop(["year", "month", "month_end"], axis=1, inplace=True) 

        return df_monthly 
      
    def _cc_rt_spot(self, base_sym, quote_sym): 

        url = f"""https://min-api.cryptocompare.com/data/price?fsym={base_sym}&tsyms={quote_sym}&api_key={self._key}""" 
        logging.info(f"url- {url}") 
        tmp = requests.get(url).json() 
        return tmp 
      
    def cc_spot(self, base_sym, quote_sym): 
        return self._cc_rt_spot(base_sym, quote_sym)