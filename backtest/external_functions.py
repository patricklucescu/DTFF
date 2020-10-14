##### DTFF PROJECT ######
## Authors:
##      - Santiago Walliser
##      - Björn Bloch
##      - Patrick Lucescu

# ----- IMPORT LIBRARIES -----
import sqlalchemy as db
import pandas as pd
import numpy as np


def get_data_from_table(_table_name, _engine):
    connection = _engine.connect()
    metadata = db.MetaData()
    data_sqlalchemy_table_obj = db.Table(_table_name, metadata, autoload=True, autoload_with=_engine)
    stmt_sqlal_obj = db.select([data_sqlalchemy_table_obj])
    exec_stmt_sqlal_obj = connection.execute(stmt_sqlal_obj)
    results = exec_stmt_sqlal_obj.fetchall()
    results = pd.DataFrame(results)
    results.columns = exec_stmt_sqlal_obj.keys()

    return results

def get_raffinot_weights(data, linkage_method='ward'):
    # Data must be a matrix where each column are returns (no date column! except it's an index column)
    # All available methods are described here:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage
    num_assets = data.shape[1]
    corr_distance = np.sqrt(2 * (1 - data.corr()))
    corr_distance_squareform_matrix = squareform(corr_distance)
    Z = linkage(corr_distance_squareform_matrix, linkage_method)

    # dn = dendrogram(Z)
    # plt.show()

    # Create a matrix, s.t. we have for each node its two neighbors
    num_clusters = len(Z) * 2
    list_of_neighbors = [None] * (num_clusters + 1)  # +1 due to zero index
    for node_index in range(len(list_of_neighbors)):
        #  Check for leave
        if node_index < num_assets:  #  -1 due to zero index
            continue  # the leaves are our assets and they do not have any neighbors

        # if we arrive first time here then node_index = num_assets
        #  Remember: At the i-th iteration, clusters with indices Z[i, 0]
        #           and Z[i, 1] are combined to form cluster n + i.
        child_nodes = Z[node_index - num_assets][:2]
        list_of_neighbors[node_index] = list(map(int, child_nodes.tolist()))

    # Assign now weights to nodes
    node_weights = [0] * (num_clusters + 1)
    node_weights[-1] = 1
    for node_index in reversed(range(len(list_of_neighbors))):
        if node_index < num_assets:  #  -1 due to zero index
            break
        weight_to_split = node_weights[node_index]
        # left child
        node_child_left_index = list_of_neighbors[node_index][0]
        node_weights[node_child_left_index] = weight_to_split / 2
        # right child
        node_child_right_index = list_of_neighbors[node_index][1]
        node_weights[node_child_right_index] = weight_to_split / 2

    # Get final weights
    asset_weights = node_weights[:num_assets]

    # Adjust
    # asset_weights = asset_weights / 100

    # last check before returning
    assert (np.array(asset_weights).sum() == 1)

    return asset_weights
pass

def getIVP(cov):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp = ivp / ivp.sum()
    return ivp
pass

def get_weights(data, date, method='equal', linkage_method='ward'):
    # TODO: Adjust for NA columns
    if method == 'equal':
        return pd.DataFrame(1 / data.shape[1], index=[date], columns=list(data.columns))
    elif method == 'raffinot':
        return pd.DataFrame(get_raffinot_weights(data, linkage_method), index=list(data.columns), columns=[date]).transpose()
    elif method == 'ivp':
        return pd.DataFrame(getIVP(data.cov()), index=list(data.columns), columns=[date]).transpose()
    else:
        raise Exception('Unknown weight method.')
pass

def compute_turnover_period(weights_pre, weights_now, returns_now, rf_rate_now = 0):
    Rp = (weights_pre * returns_now.values).sum() + (1-weights_now.sum()) * rf_rate_now
    valuePerAsset = weights_pre * (1+returns_now).values
    currentWeights = valuePerAsset / (1+Rp)
    turnover = sum(abs(weights_now - currentWeights))
    return turnover
pass

def get_tot_returns_cum(adj_returns, adj_weights, LOOKBACK=1):
    tot_returns = (adj_returns * adj_weights.values).sum(axis=1)[LOOKBACK - 1:] # -1 as index starts with 0.
    if tot_returns.iloc[0] != 0:
        tot_returns[0] = 0
    tot_returns_cum = np.cumprod(1 + tot_returns)
    return tot_returns, tot_returns_cum
pass

def compute_turnover(trading_period_dates, adj_weights, returns):
    turnover = pd.Series(0, index=trading_period_dates)
    for date_index, date in enumerate(trading_period_dates):
        if date_index == 0 or date_index == len(trading_period_dates) - 1:
            continue
        pre_date = trading_period_dates[date_index - 1]
        turnover.loc[date] = compute_turnover_period(adj_weights.loc[pre_date], adj_weights.loc[date],
                                                     returns.loc[pre_date])
    return turnover
pass

def get_tot_returns_cum_TC(tot_returns, trading_period_dates, adj_weights, returns, COSTS=0):
    turnover = compute_turnover(trading_period_dates, adj_weights, returns)
    transaction_costs = turnover * COSTS
    tot_returns_TC = tot_returns - transaction_costs
    tot_returns_TC_cum = np.cumprod(1 + tot_returns_TC)
    return tot_returns_TC, tot_returns_TC_cum
pass

def get_risk_exposure_adjusted_weights(weights, bool_list_selection, low_risk_env=False, FACTOR=1.5):
    # Get selected assets and increasse exposure
    weights.loc[:, bool_list_selection] = weights.loc[:, bool_list_selection] * FACTOR
    # If low_risk_env we do not invest in cash
    if low_risk_env:
        weights['Cash CHF'] = 0
    # Readjust weights, s.t. we have exposure of 100%
    weights = weights.div(weights.sum(axis=1).values[0])
    return weights
pass