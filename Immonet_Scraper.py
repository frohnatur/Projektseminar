from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import re

class ScrapedRealEstate():
    def __init__(self, immo_url, ort, plz, angebotspreis, grundstuecksflaeche, wohnflaeche, anzahl_zimmer, anzahl_schlafzimmer, anzahl_badezimmer, gaeste_wc, baujahr, anzahl_parkplatz, immobilienart, immobilienzustand, balkon, terrasse, heizung, befeuerungsart, etagen, geschoss, unterkellert, vermietet, energie_verbrauch, energie_effizienzklasse, aufzug, barrierefrei, denkmalschutz):
        self.immo_url = immo_url
        self.ort = ort
        self.plz = plz
        self.angebotspreis = angebotspreis
        self.grundstuecksflaeche = grundstuecksflaeche
        self.wohnflaeche = wohnflaeche
        self.anzahl_zimmer = anzahl_zimmer
        self.anzahl_schlafzimmer = anzahl_schlafzimmer
        self.anzahl_badezimmer = anzahl_badezimmer
        self.gaeste_wc= gaeste_wc
        self.baujahr = baujahr
        self.anzahl_parkplatz = anzahl_parkplatz
        self.immobilienart = immobilienart
        self.immobilienzustand = immobilienzustand
        self.balkon = balkon
        self.terrasse = terrasse
        self.heizung = heizung
        self.befeuerungsart = befeuerungsart
        self.etagen = etagen
        self.geschoss = geschoss
        self.unterkellert = unterkellert
        self.vermietet = vermietet
        self.energie_verbrauch = energie_verbrauch
        self.energie_effizienzklasse = energie_effizienzklasse
        self.aufzug = aufzug
        self.barrierefrei = barrierefrei
        self.denkmalschutz = denkmalschutz


class ImmoFetcher():
    def fetch(self):
        url = 'https://www.immonet.de/immobiliensuche/sel.do?pageoffset=1&listsize=26&objecttype=1&locationname=W%C3%BCrzburg&acid=&actype=&city=153145&ajaxIsRadiusActive=true&sortby=16&suchart=2&radius=50&pcatmtypes=-1_1&pCatMTypeStoragefield=-1_1&parentcat=-1&marketingtype=1&fromprice=&toprice=&fromarea=&toarea=&fromplotarea=&toplotarea=&fromrooms=&torooms=&wbs=-1&fromyear=&toyear=&fulltext=&absenden=Ergebnisse+anzeigen'
        inventory = []

        while url != '':
            print(url)
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            containers = soup.find_all('div', class_='flex-grow-1 display-flex flex-direction-column box-25 overflow-hidden cursor-hand')
            for container in containers:

                moredata = container.find('a', class_='block ellipsis text-225 text-default')
                immo_url = moredata.attrs['href']
                immo_url = urljoin(url, immo_url)
                print('immo_url: ' + immo_url)

                v = requests.get(immo_url)
                soup2 = BeautifulSoup(v.text, 'html.parser')

                ort = soup2.find('p', class_='text-100 pull-left')
                if ort != None:
                    vorsilbe = soup2.find('p', class_='text-100 pull-left').text.strip().split()[-5]
                    if vorsilbe == 'Bad':
                        ort = 'Bad ' + soup2.find('p', class_='text-100 pull-left').text.strip().split()[-4]
                    elif vorsilbe == 'Markt':
                        ort = 'Markt ' + soup2.find('p', class_='text-100 pull-left').text.strip().split()[-4]
                    else:
                        ort = soup2.find('p', class_='text-100 pull-left').text.strip().split()[-4]
                else:
                    ort = ''
                print('ort: ' + ort)


                plz = soup2.find('p', class_='text-100 pull-left')
                if plz != None:
                    plz = soup2.find('p', class_='text-100 pull-left').text.strip()
                    plz = re.findall(r"\d{5}", plz)[0]
                else:
                    plz = ''
                print('postleitzahl: ' + plz)


                angebotspreis = soup2.find('div', id='priceid_1')
                if angebotspreis != None:
                    angebotspreis = soup2.find('div', id='priceid_1').text.strip().split()[0]
                else:
                    angebotspreis = ''
                print('angebotspreis in €: ' + angebotspreis)


                grundstuecksflaeche = soup2.find('div', id='areaid_3')
                if grundstuecksflaeche != None:
                    grundstuecksflaeche = soup2.find('div', id='areaid_3').text.strip().split()[0]
                else:
                    grundstuecksflaeche = '0'
                print('grundstücksfläche in m²: ' + grundstuecksflaeche)


                wohnflaeche = soup2.find('div', id='areaid_1')
                if wohnflaeche != None:
                    wohnflaeche = soup2.find('div', id='areaid_1').text.strip().split()[0]
                else:
                    wohnflaeche = '0'
                print('wohnflaeche in m²: ' + wohnflaeche)


                anzahl_zimmer = soup2.find('div', id='equipmentid_1')
                if anzahl_zimmer != None:
                    anzahl_zimmer = soup2.find('div', id='equipmentid_1').text.strip().split()[0]
                else:
                    anzahl_zimmer = '0'
                print('zimmer_anzahl: ' + anzahl_zimmer)


                anzahl_schlafzimmer = soup2.find('p', id='otherDescription')
                if anzahl_schlafzimmer != None:
                    anzahl_schlafzimmer = soup2.find('p', id='otherDescription').text.strip().split()
                    if 'Schlafzimmer:' in anzahl_schlafzimmer:
                        index_beginn = anzahl_schlafzimmer.index('Schlafzimmer:')
                        anzahl_schlafzimmer1 = soup2.find('p', id='otherDescription').text.strip().split()[
                                               index_beginn:]
                        anzahl_schlafzimmer = anzahl_schlafzimmer1[1][0]
                    else:
                        anzahl_schlafzimmer = '0'

                else:
                    anzahl_schlafzimmer = '0'
                print('anzahlschlafzimmer: ' + anzahl_schlafzimmer)


                anzahl_badezimmer = soup2.find('p', id='otherDescription')
                if anzahl_badezimmer != None:
                    anzahl_badezimmer = soup2.find('p', id='otherDescription').text.strip().split()
                    if 'Badezimmer:' in anzahl_badezimmer:
                        index_beginn = anzahl_badezimmer.index('Badezimmer:')
                        anzahl_badezimmer = soup2.find('p', id='otherDescription').text.strip().split()[index_beginn:]
                        anzahl_badezimmer = anzahl_badezimmer[1][0]
                    else:
                        anzahl_badezimmer = '0'

                elif soup2.find('li', id='featureId_35') != None:
                    anzahl_badezimmer = '1'

                else:
                    anzahl_badezimmer = '0'
                print('anzahl_badezimmer: ' + anzahl_badezimmer)


                gaeste_wc = soup2.find('li', id='featureId_70')
                if gaeste_wc != None:
                    gaeste_wc = 'JA'
                else:
                    gaeste_wc = 'NEIN'
                print('gaeste_wc: ' + gaeste_wc)


                baujahr = soup2.find('div', id='yearbuild')
                if baujahr != None:
                    baujahr = soup2.find('div', id='yearbuild').text.strip().split()[0]
                else:
                    baujahr = ''
                print('baujahr: ' + baujahr)


                anzahl_parkplatz = soup2.find('div', id='equipmentid_13')
                if anzahl_parkplatz != None:
                    anzahl_parkplatz = anzahl_parkplatz = soup2.find('div', id='equipmentid_13').text.strip()
                else:
                    anzahl_parkplatz = '0'
                print('anzahl_parkplatz: ' + anzahl_parkplatz)


                immobilienart = soup2.find('h2', id='sub-headline-expose')
                if immobilienart != None:
                    immobilienart = soup2.find('h2', id='sub-headline-expose').text.strip().split()[0]
                    if immobilienart == 'Besondere':
                        immobilienart = 'Besondere Immobilie'
                else:
                    immobilienart = 'Sonstiges'
                print('immobilienart ' + immobilienart)


                immobilienzustand = soup2.find('div', id='objectstatecategoryValue')
                if immobilienzustand != None:
                    immobilienzustand = soup2.find('div', id='objectstatecategoryValue').text.strip()
                else:
                    immobilienzustand = ''
                print('immobilienzustand: ' + immobilienzustand)


                balkon = soup2.find('li', id='featureId_57')
                if balkon != None:
                    balkon = 'JA'
                else:
                    balkon = 'NEIN'
                print('balkon: ' + balkon)


                terrasse = soup2.find('li', id='featureId_67')
                if terrasse != None:
                    terrasse = 'JA'
                else:
                    terrasse = 'NEIN'
                print('terasse: ' + terrasse)


                heizung = soup2.find('div', id='heatTypeValue')
                if heizung != None:
                    heizung = soup2.find('div', id='heatTypeValue').text.strip()
                else:
                    heizung = ''
                print('heizung: ' + heizung)


                befeuerungsart = soup2.find('div', id='heaterSupplierValue')
                if befeuerungsart != None:
                    befeuerungsart = soup2.find('div', id='heaterSupplierValue').text.strip().split()
                    if len(befeuerungsart) == 2:
                        befeuerungsart = befeuerungsart[0] + ' ' + befeuerungsart[1]
                    else:
                        befeuerungsart = befeuerungsart[0]

                else:
                    befeuerungsart = ''
                print('befeuerungsart: ' + befeuerungsart)


                etagen = soup2.find('li', id='featureId_135')
                if etagen != None:
                    etagen = soup2.find('li', id='featureId_135').text.strip().split()[-1]
                else:
                    etagen = '0'
                print('etagen: ' + etagen)


                geschoss = soup2.find('li', id='featureId_123')
                if geschoss != None:
                    geschoss = soup2.find('li', id='featureId_123').text.strip().split()[-1]
                elif immobilienart == 'Erdgeschosswohnung':
                    geschoss = '0'
                else:
                    geschoss = ''
                print('geschoss: ' + geschoss)


                unterkellert = soup2.find('li', id='featureId_69')
                if unterkellert != None:
                    unterkellert = 'JA'
                else:
                    unterkellert = 'NEIN'
                print('unterkellert: ' + unterkellert)


                vermietet = soup2.find('li', id='featureId_318')
                if vermietet != None:
                    vermietet = 'JA'
                else:
                    vermietet = 'NEIN'
                print('vermietet: ' + vermietet)


                energie_verbrauch = soup2.find('div', id='energyValue')
                if energie_verbrauch != None:
                    energie_verbrauch = soup2.find('div', id='energyValue').text.strip().split()[0]
                else:
                    energie_verbrauch = ''
                print('energie_verbrauch: ' + energie_verbrauch)


                energie_effizienzklasse = soup2.find('div', id='efficiencyValue')
                if energie_effizienzklasse != None:
                    energie_effizienzklasse = soup2.find('div', id='efficiencyValue').text.strip().split()[-1]
                else:
                    energie_effizienzklasse = ''
                print('energie_effizienzklasse: ' + energie_effizienzklasse)


                aufzug = soup2.find('li', id='featureId_68')
                aufz_sonstiges = soup2.find('p', id='otherDescription')
                aufz_objektbeschreibung = soup2.find('p', id='objectDescription')

                if aufzug != None:
                    aufzug = 'JA'

                elif aufz_sonstiges != None:
                    aufz_sonstiges = soup2.find('p', id='otherDescription').text.strip()
                    aufz_sonstiges = re.findall(r"[\s\W]([Aa]ufzug[\w*\W*])", aufz_sonstiges)
                    if aufz_sonstiges != []:
                        aufzug = 'JA'

                    elif aufz_objektbeschreibung != None:
                        aufz_objektbeschreibung = soup2.find('p', id='objectDescription').text.strip()
                        aufz_objektbeschreibung = re.findall(r"[\s\W]([Aa]ufzug[\w*\W*])", aufz_objektbeschreibung)
                        if aufz_objektbeschreibung != []:
                            aufzug = 'JA'
                        else:
                            aufzug = 'NEIN'

                    else:
                        aufzug = 'NEIN'

                else:
                    aufzug = 'NEIN'
                print('aufzug: ' + aufzug)


                barrierefrei = soup2.find('li', id='featureId_167')
                barrierefrei2 = soup2.find('li', id='featureId_285')
                ba_ausstattung = soup2.find('div', id='ausstattung')
                ba_objektbeschreibung = soup2.find('p', id='objectDescription')
                ba_sonstiges = soup2.find('p', id='otherDescription')

                if barrierefrei != None:
                    barrierefrei = 'JA'

                elif barrierefrei2 != None:
                    barrierefrei = 'JA'

                elif ba_ausstattung != None:
                    ba_ausstattung = soup2.find('div', id='ausstattung').text.strip()
                    ba_ausstattung = re.findall(r"[\s\W]([Bb]arrierefrei[\w*\W*])", ba_ausstattung)
                    if ba_ausstattung != []:
                        barrierefrei = 'JA'

                    elif ba_objektbeschreibung != None:
                        ba_objektbeschreibung = soup2.find('p', id='objectDescription').text.strip()
                        ba_objektbeschreibung = re.findall(r"[\s\W]([Bb]arrierefrei[\w*\W*])", ba_objektbeschreibung)
                        if ba_objektbeschreibung != []:
                            barrierefrei = 'JA'

                        elif ba_sonstiges != None:
                            ba_sonstiges = soup2.find('p', id='otherDescription').text.strip()
                            ba_sonstiges = re.findall(r"[\s\W]([Bb]arrierefrei[\w*\W*])", ba_sonstiges)
                            if ba_sonstiges != []:
                                barrierefrei = 'JA'
                            else:
                                barrierefrei = 'NEIN'

                        else:
                            barrierefrei = 'NEIN'

                    else:
                        barrierefrei = 'NEIN'

                else:
                    barrierefrei = 'NEIN'

                print('barrierefrei: ' + barrierefrei)


                denkmalschutz = soup2.find('p', id='objectDescription')
                denkm_objektbeschreibung = soup2.find('p', id='objectDescription')
                denkm_sonstiges = soup2.find('p', id='otherDescription')
                if denkm_objektbeschreibung != None:
                    denkm_objektbeschreibung = soup2.find('p', id='objectDescription').text.strip()
                    denkm_objektbeschreibung = re.findall(r"\w*\s*\w*(?=\s*\W*[Dd]enkmal\w*)", denkm_objektbeschreibung)
                    if denkm_objektbeschreibung != []:
                        denkm_kein = re.findall(r"\s*\W*\w*([Kk]ein)", denkm_objektbeschreibung[0].strip())
                        denkma_nichtunter = re.findall(r"\s*(nicht\sunter)", denkm_objektbeschreibung[0].strip())
                        if denkm_kein != []:
                            denkmalschutz = 'NEIN'
                        elif denkma_nichtunter != []:
                            denkmalschutz = 'NEIN'
                        else:
                            denkmalschutz = 'JA'
                    else:
                        denkmalschutz = 'NEIN'

                elif denkm_sonstiges != None:
                    denkm_sonstiges = soup2.find('p', id='otherDescription').text.strip()
                    denkm_sonstiges = re.findall(r"\w*\s*\w*(?=\s*\W*[Dd]enkmal\w*)", denkm_sonstiges)
                    if denkm_sonstiges != []:
                        denkm_kein = re.findall(r"\s*\W*\w*([Kk]ein)", denkm_sonstiges[0].strip())
                        denkma_nichtunter = re.findall(r"\s*(nicht\sunter)", denkm_sonstiges[0].strip())
                        if denkm_kein != []:
                            denkmalschutz = 'NEIN'
                        elif denkma_nichtunter != []:
                            denkmalschutz = 'NEIN'
                        else:
                            denkmalschutz = 'JA'
                    else:
                        denkmalschutz = 'NEIN'

                else:
                    denkmalschutz = 'NEIN'

                print('denkmalschutz: ' + denkmalschutz)


                scraped = ScrapedRealEstate(immo_url, ort, plz, angebotspreis, grundstuecksflaeche, wohnflaeche,
                                            anzahl_zimmer, anzahl_schlafzimmer, anzahl_badezimmer, gaeste_wc, baujahr,
                                            anzahl_parkplatz, immobilienart, immobilienzustand, balkon, terrasse,
                                            heizung, befeuerungsart, etagen, geschoss, unterkellert, vermietet,
                                            energie_verbrauch, energie_effizienzklasse, aufzug, barrierefrei,
                                            denkmalschutz)
                inventory.append(scraped)


            next_button = soup.find('a', class_='col-sm-3 col-xs-1 pull-right text-right')
            if next_button:
                next_href = next_button.attrs["href"]
                next_href = urljoin(url, next_href)
                url = next_href
            else:
                url = ''
        return inventory



import csv

fetcher = ImmoFetcher()

with open('immonet_komplett.csv', 'w', newline='', encoding='utf-8') as csvfile:
    realestatewriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    realestatewriter.writerow(
        ['immo_url', 'ort', 'plz', 'angebotspreis', 'grundstuecksflaeche', 'wohnflaeche', 'anzahl_zimmer',
         'anzahl_schlafzimmer', 'anzahl_badezimmer', 'gaeste_wc', 'baujahr', 'anzahl_parkplatz', 'immobilienart',
         'immobilienzustand', 'balkon', 'terrasse', 'heizung', 'befeuerungsart', 'etagen', 'geschoss', 'unterkellert',
         'vermietet', 'energie_verbrauch', 'energie_effizienzklasse', 'aufzug', 'barrierefrei', 'denkmalschutz'])

    for x in fetcher.fetch():
        realestatewriter.writerow(
            [x.immo_url, x.ort, x.plz, x.angebotspreis, x.grundstuecksflaeche, x.wohnflaeche, x.anzahl_zimmer,
             x.anzahl_schlafzimmer, x.anzahl_badezimmer, x.gaeste_wc, x.baujahr, x.anzahl_parkplatz, x.immobilienart,
             x.immobilienzustand, x.balkon, x.terrasse, x.heizung, x.befeuerungsart, x.etagen, x.geschoss,
             x.unterkellert, x.vermietet, x.energie_verbrauch, x.energie_effizienzklasse, x.aufzug, x.barrierefrei,
             x.denkmalschutz])