import pandas as pd
import numpy as np

if __name__ == "__main__":
    #Datensatz einlesen
    ImmobilienMaster = pd.read_excel(r"Files/ImmobilienAll2v3.xlsx", index_col="Unnamed: 0")

    #Tausender Stellen - Scraper Fehler -> abgeschnittene Nullen korrigieren
    ImmobilienMaster.loc[ImmobilienMaster["angebotspreis"] <= 10000, "angebotspreis"] = ImmobilienMaster["angebotspreis"] * 1000

    #Umbenennungen
    ImmobilienMaster.rename(columns={"befeuerungsart" : "energietyp"}, inplace=True)

    #Zeilen ohne Angebotspreis und nutzlose Spalten droppen
    ImmobilienMaster = ImmobilienMaster.dropna(subset=["angebotspreis"])
    ImmobilienMaster = ImmobilienMaster.drop(columns=['bezugsfrei ab', "denkmalschutzobjekt", "einbauküche", "immo_url",
                                   "energieausweis", "energie­ausweistyp", "fahrstuhl", "grundbucheintrag",
                                   "grunderwerbssteuer", 'hausgeld', "maklerprovision",
                                   "modernisierung/ sanierung", "monatsmiete", "notarkosten", "ort", "scoutid",
                                   "strasse", "web-scraper-start-url", "wohnung-href",
                                   "denkmalschutz", "nutzfläche ca", "ausstattung", "ausstattung beschreibung", "lage",
                                    "objektbeschreibung", "sonstiges", "wohnung"])

    #Spaltentypen bearbeiten
    ImmobilienMaster["balkon"] = ImmobilienMaster["balkon"].astype("category")
    ImmobilienMaster["barrierefrei"] = ImmobilienMaster["barrierefrei"].astype("category")
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].astype("category")
    ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].astype("category")
    ImmobilienMaster["gaeste_wc"] = ImmobilienMaster["gaeste_wc"].astype("category")
    ImmobilienMaster["heizung"] = ImmobilienMaster["heizung"].astype("category")
    ImmobilienMaster["immobilienart"] = ImmobilienMaster["immobilienart"].astype("category")
    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].astype("category")
    ImmobilienMaster["plz"] = ImmobilienMaster["plz"].astype("category")
    ImmobilienMaster["terrasse"] = ImmobilienMaster["terrasse"].astype("category")
    ImmobilienMaster["unterkellert"] = ImmobilienMaster["unterkellert"].astype("category")
    ImmobilienMaster["vermietet"] = ImmobilienMaster["vermietet"].astype("category")
    ImmobilienMaster["aufzug"] = ImmobilienMaster["aufzug"].astype("category")

    #Einzelne Spaltenwerte ausbessern
    ImmobilienMaster.loc[ImmobilienMaster["aufzug"].isna(), "aufzug"] = "NEIN"

    #Imputation -- DUMMY Modus
    #Anzahl Badezimmer: Annahme, dass wenn es sich nicht um ein leeres Grundstück handelt gibt es mindestens 1 Badezimmer
    ImmobilienMaster.loc[(ImmobilienMaster["anzahl_badezimmer"] == 0) & (ImmobilienMaster["immobilienart"] != "Wohngrundstück"), "anzahl_badezimmer"] = 1
    ImmobilienMaster["anzahl_badezimmer"] = ImmobilienMaster["anzahl_badezimmer"].fillna(1)

    # Anzahl Schlafzimmer: Annahme, dass wenn es sich nicht um ein leeres Grundstück handelt gibt es mindestens 1 Schlafzimmer
    ImmobilienMaster.loc[(ImmobilienMaster["anzahl_schlafzimmer"] == 0) & (ImmobilienMaster["immobilienart"] != "Wohngrundstück"), "anzahl_schlafzimmer"] = 1
    ImmobilienMaster["anzahl_schlafzimmer"] = ImmobilienMaster["anzahl_schlafzimmer"].fillna(1)

    # Anzahl Zimmer: Annahme, dass wenn es sich nicht um ein leeres Grundstück handelt gibt es mindestens 1 Zimmer gibt
    ImmobilienMaster.loc[(ImmobilienMaster["anzahl_zimmer"] == 0) & (ImmobilienMaster["immobilienart"] != "Wohngrundstück"), "anzahl_zimmer"] = 1




    #Ausgabe
    print(ImmobilienMaster.info())
    print(ImmobilienMaster["geschoss"].value_counts())
    print(ImmobilienMaster["etagen"].value_counts())


    ImmobilienMaster.to_excel(excel_writer="Files/ImmobilienMasterV2.xlsx", sheet_name="ImmobilienAll")

    # Display Optionen für Konsole
    # with pd.option_context('display.max_rows', 5, 'display.max_columns', 17):
     #   print(ImmobilienMaster)
