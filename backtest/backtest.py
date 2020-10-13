##### DTFF PROJETC ######
## Authors:
##      - Santiago Walliser
##      - Björn Bloch


# ----- IMPORT LIBRARIES -----
import sys
import numpy as np
import os
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import pandas as pd

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

