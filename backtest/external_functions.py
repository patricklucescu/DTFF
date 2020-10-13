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