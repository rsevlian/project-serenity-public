import datetime 
  
# samplerate flags. 
SAMPLE_RATE_WEEKLY = 52 
SAMPLE_RATE_DAILY = 365 
SAMPLE_RATE_12HOUR = 1460 
SAMPLE_RATE_6HOUR = 1460 
SAMPLE_RATE_HOURLY = 8760 
  
# Time Constants 
DAYS_IN_WEEK = 7 
HOURS_IN_DAY = 24 
WEEK_IN_YEAR = 52 
DAYS_IN_YEAR = 365 
  
# Halving Date 
HALVING_0 = datetime.date(2012, 11, 28) 
HALVING_1 = datetime.date(2016, 7, 9) 
HALVING_2 = datetime.date(2020, 5, 11) 
HALVING_3 = datetime.date(2024, 2, 13)  # approx 
HALVING_4 = datetime.date(2028, 1, 1)   # approx 
HALVING_5 = datetime.date(2032, 1, 1)   # approx 
HALVING_6 = datetime.date(2036, 1, 1)   # approx 
HALVING_7 = datetime.date(2042, 1, 1)   # approx 
  
# Daliy Newtwork Block Reward (approx. 10 minutes per block) 
HALVING_BTC_0 = 24 * 6 * 50 * (0.5 ** 0) 
HALVING_BTC_1 = 24 * 6 * 50 * (0.5 ** 1) 
HALVING_BTC_2 = 24 * 6 * 50 * (0.5 ** 2) 
HALVING_BTC_3 = 24 * 6 * 50 * (0.5 ** 3) 
HALVING_BTC_4 = 24 * 6 * 50 * (0.5 ** 4) 
HALVING_BTC_5 = 24 * 6 * 50 * (0.5 ** 5) 
HALVING_BTC_6 = 24 * 6 * 50 * (0.5 ** 6) 
HALVING_BTC_7 = 24 * 6 * 50 * (0.5 ** 7) 
HALVING_BTC_8 = 24 * 6 * 50 * (0.5 ** 8) 
  
# China Crackdown Date 
CHINA_CRACKDOWN_START = datetime.date(2021, 5, 14) 
CHINA_CRACKDOWN_END = datetime.date(2021, 7, 1) 
  
# Regression Line Segments 
LM_S1 = datetime.date(2019, 6, 1) 
LM_E1 = CHINA_CRACKDOWN_START 
LM_S2 = datetime.date(2022, 1, 1) 
# LM_E2 will be the current time. 
  
# machine price file local. 
LUXOR_MACHINE_PRICE_FILE_1 = "data/true_machine_pricing.csv" 
LUXOR_MACHINE_PRICE_FILE_2 = "data/machine_clean_1.csv" 
  
ASSET_BTC = "BTC" 
ASSET_USD = "USD" 
  
MACHINE_S17 = "S17" 
MACHINE_S19J = "S19J" 
MACHINE_M20 = "M20" 
MACHINE_M30 = "M30" 
MACHINE_S9 = "S9"

CONTRACT_CALL_OPTION = "CONTRACT_CALL_OPTION" 
CONTRACT_PUT_OPTION = "CONTRACT_PUT_OPTION" 
  
DIRECTION_LONG_CONTRACT  = "DIRECTION_LONG"
DIRECTION_SHORT_CONTRACT = "DIRECTION_SHORT"
  
def machine_params(): 
  
    dct = { 
        "S19J": { 
            "energy_consumption": 3.5, 
            "hash_rate": 100, 
        }, 
        "S9": { 
            "energy_consumption": 1.31, 
            "hash_rate": 13 
        }, 
        "S17": { 
            "energy_consumption": 2.288, 
            "hash_rate": 64 
        }, 
        "M20": { 
            "energy_consumption": 3.36, 
            "hash_rate": 68 
        }, 
        "M30": { 
            "energy_consumption": 3.268, 
            "hash_rate": 86 
        } 
    } 
  
    return dct 
  
