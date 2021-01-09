import pandas as pd
import numpy as np

if __name__ == "__main__":
    #Datensatz einlesen
    ImmobilienMaster = pd.read_excel(r"Files/ImmobilienAll2v3.xlsx", index_col="Unnamed: 0")
    #1000000 scraper fehler -> abgeschnittene Nullen korrigieren
    ImmobilienMaster.loc[ImmobilienMaster["angebotspreis"] <= 10000, "angebotspreis"] = ImmobilienMaster["angebotspreis"] * 1000
    ImmobilienMaster = ImmobilienMaster.dropna(subset=["angebotspreis"])
    ImmobilienMaster["plz"].astype(object)

    ImmobilienMaster.rename(columns={"befeuerungsart" : "energietyp"}, inplace=True)
    ImmobilienMaster = ImmobilienMaster.drop(columns=['bezugsfrei ab', "denkmalschutzobjekt", "einbauküche", "immo_url",
                                   "energieausweis", "energie­ausweistyp", "fahrstuhl", "grundbucheintrag",
                                   "grunderwerbssteuer", 'hausgeld', "maklerprovision",
                                   "modernisierung/ sanierung", "monatsmiete", "notarkosten", "ort", "scoutid",
                                   "strasse", "web-scraper-start-url", "wohnung-href",
                                   "denkmalschutz", "nutzfläche ca", "ausstattung", "ausstattung beschreibung", "lage",
                                    "objektbeschreibung", "sonstiges", "wohnung"])

    print(ImmobilienMaster.info())
    ImmobilienMaster.to_excel(excel_writer="Files/ImmobilienMasterV2.xlsx", sheet_name="ImmobilienAll")

   # with pd.option_context('display.max_rows', 5, 'display.max_columns', 17):
     #   print(ImmobilienMaster)


#to do
#garage/stellplatz dummies
#wohnung oder haus -> neue spalte