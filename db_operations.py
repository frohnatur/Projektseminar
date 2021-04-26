import pickle

import pandas as pd
import sqlite3
from sqlite3 import Error

import main
db_connection = sqlite3.connect('Datenbank/ImmoDB.db')
#input_df = pd.read_sql_query('SELECT * FROM Features', con=db_connection, index_col="index")
#load_XGB_modell = pickle.load(open('XGB_Standardmodell_20210421-1620.pckl', 'rb'))
#output = int(load_XGB_modell.predict(input_df)[0])
if __name__ == "__main__":
   print()