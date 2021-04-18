import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Metadaten = pd.read_excel(r'Files/Meta_Data/Metadaten Bayern 13.04.xlsx', skiprows=[0,1,2])
Metadaten = Metadaten.dropna(axis='columns')
Metadaten.drop(columns=["Hilfe Gemeindeschlüssel", "Hilfe Ort"], inplace=True)
#Metadaten = Metadaten.astype({" Arbeitslosenquote in Prozent": float, " Anteil nicht erfolgreicher beruflicher Bildungsgänge" : float })
Daten = pd.read_excel(r'Files/Tests/imputed_data_20210404.xlsx', index_col="Unnamed: 0")
Daten = Daten.astype({"plz":int})
Trainingsdaten = pd.merge(Daten, Metadaten, how="inner", on="plz")

if __name__ == "__main__":
   print(Metadaten.info())
   print(Daten.info())
   print(len(list(Trainingsdaten.plz.unique())))


