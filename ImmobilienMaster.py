import pandas as pd
import numpy as np

if __name__ == "__main__":
    #Datensatz einlesen
    ImmobilienMaster = pd.read_excel(r"Files/ImmobilienAll2v2.xlsx", index_col="Unnamed: 0")
    #1000000 scraper fehler -> abgeschnittene Nullen korrigieren
    ImmobilienMaster.loc[ImmobilienMaster["angebotspreis"] <= 10000, "angebotspreis"] = ImmobilienMaster["angebotspreis"] * 1000
    ImmobilienMaster["grundstuecksflaeche"].astype(int)
    #ImmobilienMaster["wohnflaeche"].astype(float)

    ImmobilienMaster.rename(columns={"balkon" : "balkon oder terasse", "befeuerungsart" : "energietyp"}, inplace=True)
    ImmobilienMaster = ImmobilienMaster.drop(columns=['bezugsfrei ab', "denkmalschutzobjekt", "einbauküche", "immo_url",
                                   "energieausweis", "energie­ausweistyp", "fahrstuhl", "grundbucheintrag",
                                   "grunderwerbssteuer", 'hausgeld', "maklerprovision",
                                   "modernisierung/ sanierung", "monatsmiete", "notarkosten", "ort", "scoutid",
                                   "strasse", "web-scraper-start-url", "wohnung-href",
                                   "denkmalschutz"])

    ImmobilienMaster.to_excel(excel_writer="Files/ImmobilienMaster.xlsx", sheet_name="ImmobilienAll")

    with pd.option_context('display.max_rows', 5, 'display.max_columns', 17):
        print(ImmobilienMaster)


#to do
#garage/stellplatz dummies
#wohnung oder haus -> neue spalte