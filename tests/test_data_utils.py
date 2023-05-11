import sys
sys.path.append("../lib")

import data_utils

import numpy as np
import pandas as pd

def test_data_utils():

	## generate_weekly_df
	du = data_utils.DataUtility("")
	du._raw_network_price_df = pd.read_csv("../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")
	du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
	du._df_feature() 
	du._aggregate_weekly_df() 

	assert du._weekly_network_price_df.sum().sum() == 35545444113.88769

	## generate_weekly_machine_price_df
	du = data_utils.DataUtility("")
	du._raw_network_price_df = pd.read_csv("../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")
	du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
	du._df_feature() 

	machine_prices = data_utils.machine_market_prices("../data/machine_prices.csv")
	du._weekly_machine_price_df = du._join_machine_price(machine_prices, HR_ROLLING_AVG=30)

	print(du._weekly_machine_price_df.sum().sum())
	assert du._weekly_machine_price_df.sum().sum() == 20878967243.593388





