import pandas as pd
import sqlite3
import machine_learning as ml
import numpy as np
import matplotlib.pyplot as plt
db_connection = sqlite3.connect('Datenbank/ImmoDB.db')
Meta_Daten_Beispiel = pd.read_sql_query('SELECT * FROM Meta_Data WHERE plz=97070', con=db_connection, index_col="index")
#Trainingsdaten = pd.read_sql_query('SELECT * FROM ML_Trainingsdaten', db_connection, index_col='index')
#Trainingsdaten.columns = Trainingsdaten.columns.str.lstrip()
#Trainingsdaten.columns = Trainingsdaten.columns.str.rstrip()
#Trainingsdaten.to_sql(name='ML_Trainingsdaten', con=db_connection, if_exists='replace')
#Trainingsdaten = ml.outlier_drop(Trainingsdaten)
#Trainingsdaten = ml.boolean(Trainingsdaten)
#Trainingsdaten = ml.variables(Trainingsdaten)
#Trainingsdaten.drop(columns=["plz"], inplace=True)
#Trainingsdaten = Trainingsdaten.astype({"angebotspreis":int})
#x_test, x_train, y_test, y_train = ml.tr_te_spl(Trainingsdaten)
#x_train_num, x_val_num = ml.numeric(x_train, x_test)
#x_train_num, x_val_num = ml.standardization(x_train_num, x_val_num)
#x_train_cat, x_val_cat = ml.category(x_train, x_test)
#x_train_target, x_val_target = ml.target_encoding(x_train_cat, x_val_cat, y_train)
#x_train, x_test = ml.joint(x_train_num, x_train_target, x_val_num, x_val_target)
#cols_train = x_train.select_dtypes(include=[np.float64]).columns
#x_train[cols_train] = x_train[cols_train].astype(np.float32)
#cols_test = x_test.select_dtypes(include=[np.float64]).columns
#x_test[cols_test] = x_test[cols_test].astype(np.float32)
#x_train.fillna(0, inplace=True)
#x_test.fillna(0, inplace=True)
#x_train = x_train.reindex(sorted(x_train.columns), axis=1)
#x_test = x_test.reindex(sorted(x_test.columns), axis=1)
#print(np.where(x_train.values >= np.finfo(np.float64).max))

#ml.ml_tests(x_train, x_test, y_train, y_test, Trainingsdaten)
#Metadaten = pd.read_excel(r'Files/Meta_Data/Metadaten Bayern 14.04.xlsx', skiprows=[0,1,2])
#Metadaten = Metadaten.dropna(axis='columns')
#Metadaten.drop(columns=["Hilfe Gemeindeschlüssel", "Hilfe Ort"], inplace=True)
#Metadaten.to_sql(name='Meta_Data', con=db_connection, if_exists='replace')
#Metadaten = Metadaten.astype({" Arbeitslosenquote in Prozent": float, " Anteil nicht erfolgreicher beruflicher Bildungsgänge" : float })
#Daten = pd.read_excel(r'Files/Tests/imputed_data_20210404.xlsx', index_col="Unnamed: 0")
#Daten = Daten.astype({"plz":int})
#Trainingsdaten = pd.merge(Daten, Metadaten, how="inner", on="plz")


if __name__ == "__main__":
 print(Meta_Daten_Beispiel)
 #print(Trainingsdaten.columns)
 #print(Trainingsdaten.info())
 #print(x_train.info())
 #print(x_test.info())
 #print(y_train.describe)

