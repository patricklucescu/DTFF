##### DTFF PROJECT ######
## Authors:
##      - Santiago Walliser
##      - Björn Bloch
##      - Patrick Lucescu


# ----- IMPORT LIBRARIES -----
import sys
import numpy as np
import os
import sqlalchemy as db
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import pandas as pd
from backtest.external_functions import *

# ----- BACKTEST SETTINGS -----
LOOKBACK = 12  # How many months should be lookback period be for the computation of correlation matrix used in Raffinot approach.
COSTS = 0.0025  # Transaction costs: 25 BPS
FACTOR = 2  # Exposure increasing factor (Lamda in paper). If Factor > 1, then we make use of News Sentiment and increase our exposure accordingly.
BOUNDARY = 0  # News Sentiment Boundary -> If e.g. 0.25, we define low (/high) risk environment if value is below -0.25 (+0.25). If in-between we have neutral NS.
START_MONTH = 0  # We start investing end of Month. If = 0 we start investing on January (2004-01-31), that is our first investment month is February.


# ----- SETUP OF BACKTEST OUTPUT FOLDER -----
if FACTOR > 1:
    BACKTEST_NAME = 'with_NS'
else:
    BACKTEST_NAME = 'without_NS'
BACKTEST_NAME = BACKTEST_NAME + '_F_' + str(FACTOR) + '_B_' + str(BOUNDARY) + '_LB_' + str(LOOKBACK) + '_' + str(START_MONTH)
# Create folder directory to save backtest output
backtest_path = os.path.join("output", BACKTEST_NAME)
if not os.path.exists(backtest_path):
    os.makedirs(backtest_path)

# ----- READ DATA -----
# Connect to database engine
engine_path = os.path.join("data", "dtff_database.sqlite3")
dtff_engine = db.create_engine('sqlite:///' + engine_path)

returns = get_data_from_table("returns", dtff_engine)
news_sentiment = get_data_from_table('Global_News_Sentiment_Macro', dtff_engine)
classification = get_data_from_table('classification_risk_or_no_risk', dtff_engine).iloc[:-1]
low_risk_env_selection = list(map(bool, classification['low_risk_environment'].tolist()))
high_risk_env_selection = list(map(bool, classification['high_risk_environment'].tolist()))

# ----- PREPROCESS DATA FOR BACKTEST -----
# Adjust return data
returns.index = returns.date
liability_proxys = returns.iloc[:, -4:]  # we dont make use of the liability_proxys variables, so we kick them out
returns = returns.iloc[:, 1:13]


# Adjust new sentiment data
news_sentiment.index = news_sentiment.date
news_sentiment = news_sentiment.avg_sentiment_class.ewm(span=30).mean()
news_sentiment = news_sentiment.shift(1).dropna()
news_sentiment.index = pd.to_datetime(news_sentiment.index)
news_sentiment = news_sentiment.resample('M').sum()

# Normalize data s.t. between 1 and -1
news_sentiment = 2*(news_sentiment - news_sentiment.min(axis=0)).div(news_sentiment.max(axis=0) - news_sentiment.min(axis=0))-1
news_sentiment.index = [pd.to_datetime(date).date() for date in list(news_sentiment.index)]  # readjust index

# Get only relevant dates
returns = returns.reindex(news_sentiment.index.tolist())

# Set needed variables
all_dates = list(returns.index)
num_assets = returns.shape[1]
investment_period = all_dates[LOOKBACK:]


# ---- BACKTEST ------
# Create weight matrices
weights_single = pd.DataFrame(0, index=all_dates, columns=list(returns.columns))
weights_complete = weights_single.copy()
weights_average = weights_single.copy()
weights_ward = weights_single.copy()
weights_ivp = weights_single.copy()
weights_eq = weights_single.copy()

# Loop over days and trade accordingly
date_index = 0
date = investment_period[0]
for date_index, date in enumerate(investment_period):

    # Rebalancing date, yes or no?
    if date.month == investment_period[START_MONTH].month:  # in [1,4,7,10]:

        print('New investment date: ' + str(date) + '', str(date_index))

        # Get returns up to investment date.
        return_window = returns.loc[:date]

        # Compute weights using different approaches
        weights_single_temp = get_weights(return_window[-LOOKBACK:], date, method='raffinot', linkage_method='single')
        weights_complete_temp = get_weights(return_window[-LOOKBACK:], date, method='raffinot', linkage_method='complete')
        weights_average_temp = get_weights(return_window[-LOOKBACK:], date, method='raffinot', linkage_method='average')
        weights_ward_temp = get_weights(return_window[-LOOKBACK:], date, method='raffinot', linkage_method='ward')
        weights_ivp_temp = get_weights(return_window[-LOOKBACK:], date, method='ivp')
        weights_eq_temp = get_weights(return_window[-LOOKBACK:], date)

        # If Factor > 1, then we make use of News Sentiment and increase our exposure accordingly.
        if news_sentiment[date] > BOUNDARY:  # low risk environment
            weights_single_temp = get_risk_exposure_adjusted_weights(weights_single_temp, low_risk_env_selection, low_risk_env=True, FACTOR=FACTOR)
            weights_complete_temp = get_risk_exposure_adjusted_weights(weights_complete_temp, low_risk_env_selection, low_risk_env=True, FACTOR=FACTOR)
            weights_average_temp = get_risk_exposure_adjusted_weights(weights_average_temp, low_risk_env_selection, low_risk_env=True, FACTOR=FACTOR)
            weights_ward_temp = get_risk_exposure_adjusted_weights(weights_ward_temp, low_risk_env_selection, low_risk_env=True, FACTOR=FACTOR)
        elif news_sentiment[date] <= (-1*BOUNDARY):  # high risk environment
            weights_single_temp = get_risk_exposure_adjusted_weights(weights_single_temp, high_risk_env_selection, FACTOR=FACTOR)
            weights_complete_temp = get_risk_exposure_adjusted_weights(weights_complete_temp, high_risk_env_selection, FACTOR=FACTOR)
            weights_average_temp = get_risk_exposure_adjusted_weights(weights_average_temp, high_risk_env_selection, FACTOR=FACTOR)
            weights_ward_temp = get_risk_exposure_adjusted_weights(weights_ward_temp, high_risk_env_selection, FACTOR=FACTOR)

        # Update weight matrix we new data row
        weights_single.update(weights_single_temp)
        weights_complete.update(weights_complete_temp)
        weights_average.update(weights_average_temp)
        weights_ward.update(weights_ward_temp)
        weights_ivp.update(weights_ivp_temp)
        weights_eq.update(weights_eq_temp)
    else:
        # No rebalancing date. Readjust weights, such that we keep fixed exposure during whole holding period.
        weights_single.loc[date] = weights_single.loc[investment_period[date_index - 1]]
        weights_complete.loc[date] = weights_complete.loc[investment_period[date_index - 1]]
        weights_average.loc[date] = weights_average.loc[investment_period[date_index - 1]]
        weights_ward.loc[date] = weights_ward.loc[investment_period[date_index-1]]
        weights_ivp.loc[date] = weights_ivp.loc[investment_period[date_index-1]]
        weights_eq.loc[date] = weights_eq.loc[investment_period[date_index - 1]]


# Adjust return frame to compute
adj_returns = returns[1:]
adj_weights_single = weights_single[:-1]
adj_weights_complete = weights_complete[:-1]
adj_weights_average = weights_average[:-1]
adj_weights_ward = weights_ward[:-1]
adj_weights_ivp = weights_ivp[:-1]
adj_weights_eq = weights_eq[:-1]

# Returns with NOTC (No Transaction Costs)
pf_tot_returns_single, pf_tot_returns_single_cum = get_tot_returns_cum(adj_returns, adj_weights_single, LOOKBACK)
pf_tot_returns_complete, pf_tot_returns_complete_cum = get_tot_returns_cum(adj_returns, adj_weights_complete, LOOKBACK)
pf_tot_returns_average, pf_tot_returns_average_cum = get_tot_returns_cum(adj_returns, adj_weights_average, LOOKBACK)
pf_tot_returns_ward, pf_tot_returns_ward_cum = get_tot_returns_cum(adj_returns, adj_weights_ward, LOOKBACK)
bm_tot_returns_ivp, bm_tot_returns_ivp_cum = get_tot_returns_cum(adj_returns, adj_weights_ivp, LOOKBACK)
bm_tot_returns_eq, bm_tot_returns_eq_cum = get_tot_returns_cum(adj_returns, adj_weights_eq, LOOKBACK)

# Returns with TCs
trading_period_dates = pf_tot_returns_ward.index
pf_tot_returns_single_TC, pf_tot_returns_single_TC_cum = get_tot_returns_cum_TC(pf_tot_returns_single, trading_period_dates, adj_weights_single, returns, COSTS)
pf_tot_returns_complete_TC, pf_tot_returns_complete_TC_cum = get_tot_returns_cum_TC(pf_tot_returns_complete, trading_period_dates, adj_weights_complete, returns, COSTS)
pf_tot_returns_average_TC, pf_tot_returns_average_TC_cum = get_tot_returns_cum_TC(pf_tot_returns_average, trading_period_dates, adj_weights_average, returns, COSTS)
pf_tot_returns_ward_TC, pf_tot_returns_ward_TC_cum = get_tot_returns_cum_TC(pf_tot_returns_ward, trading_period_dates, adj_weights_ward, returns, COSTS)
bm_tot_returns_ivp_TC, bm_tot_returns_ivp_TC_cum = get_tot_returns_cum_TC(bm_tot_returns_ivp, trading_period_dates, adj_weights_ivp, returns, COSTS)
bm_tot_returns_eq_TC, bm_tot_returns_eq_TC_cum = get_tot_returns_cum_TC(bm_tot_returns_eq, trading_period_dates, adj_weights_eq, returns, COSTS)

# Create one dataframe with all cumulative returns
tot_returns_cum_noTC_and_TC = pd.DataFrame([pf_tot_returns_single_cum, pf_tot_returns_complete_cum,
                                            pf_tot_returns_average_cum, pf_tot_returns_ward_cum,
                                            bm_tot_returns_ivp_cum, bm_tot_returns_eq_cum,
                                            pf_tot_returns_single_TC_cum, pf_tot_returns_complete_TC_cum,
                                            pf_tot_returns_average_TC_cum, pf_tot_returns_ward_TC_cum,
                                            bm_tot_returns_ivp_TC_cum, bm_tot_returns_eq_TC_cum],
                                           index=['Single', 'Complete', 'Average', 'Ward', 'BM Ivp', 'BM Eq',
                                                  'Single TC', 'Complete TC', 'Average TC', 'Ward TC', 'BM Ivp TC', 'BM Eq TC']).transpose()

# Write dataframe to db
tot_returns_cum_noTC_and_TC.to_sql(BACKTEST_NAME, dtff_engine, if_exists='replace')