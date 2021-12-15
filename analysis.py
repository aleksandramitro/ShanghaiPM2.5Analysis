# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:06:44 2021

@author: Aleksandra Mitro
"""

#%% Importovanje biblioteka za obradu i analizu podataka
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
import missingno as msno

#%%Ucitavanje baze u DataFrame

df = pd.read_csv("ShanghaiPM20100101_20151231.csv")

#%% Oblik baze - 52584 uzorka i 17 obelezja pri cemu uzorak baze pokazuje vremenske uslove u satu u toku dana

print(df.shape)
#%% Prikaz prvih 5 uzoraka iz baze
print(df.head())
#Kategoricka obelezja su year,month,day,hour,season, cbwd dok su ostala obelezja numericka
#%%Provera da li postoje nedostajajuce vrednosti
print(df.isnull().sum())
#%%Procentualni prikaz za nedostajajuce vrednosti
print(df.isnull().sum()/len(df)*100)
#Vise od 50% uzoraka ima nedostajajuce vrednosti za obelezje PM_Jingan
#Oko 35% uzoraka ima nedostajajuce vrednosti za obelezje PM_US Post
#Vise od 50% uzoraka ima nedostajajuce vrednosti za obelezje PM_Xuhui
#Dok za obelezja DEWP,HUMI,PRES,TEMP,cbwd,lsw manje od 0.1% uzoraka ima nedostajajuce vrednosti
#Za obelezja precipitation i lprec oko 7% uzoraka ima nedostajajucu vrednost
#%% Grupisanje po godinama kako bi videli broj nedostajajucih vrednosti za svaku godinu
gbyNanYear = df.groupby(["year"]).apply(lambda x : x.isnull().sum().sum()).reset_index()
# Najvise nedostajajucih vrednosti imamo u 2010 godini i trend sa puno nedostajajucih vrednosti
# se nastavlja i u 2011 i 2012 godini dok se taj broj znacajno smanjio u 2013 godini 
#%% Graficki prikaz broja nedostajajucih vrednosti za svako obelezje
msno.bar(df)
#%% Heatmap za nedostajajuce vrednosti za svako obelezje
msno.heatmap(df)

#%% Izbacivanje obelezja koja se ne odnose na merenja u PM_US Post-u
df.drop(['PM_Jingan','PM_Xuhui'], inplace=True, axis=1)
#%%Izbacujem uzorke koji imaju vrednost nan za obelezja DEWP,HUMI,PRES,TEMP,cbwd,lws
indexOfNanValuesInDEWP = df.loc[df['DEWP'].isnull()].index
#%%
df.drop(indexOfNanValuesInDEWP, inplace= True, axis = 0)
#%%
indexOfNanValuesInHUMI = df.loc[df['HUMI'].isnull()].index
#%%
df.drop(indexOfNanValuesInDEWP, inplace= True, axis = 0)
#%%
indexOfNanValuesInPRES = df.loc[df['PRES'].isnull()].index
#%%
df.drop(indexOfNanValuesInPRES, inplace= True, axis = 0)
#%%
indexOfNanValuesInCBWD = df.loc[df['cbwd'].isnull()].index
#%%
df.drop(indexOfNanValuesInCBWD, inplace= True, axis = 0)
#%%
indexOfNanValuesInIWS = df.loc[df['Iws'].isnull()].index
#%%
df.drop(indexOfNanValuesInIWS, inplace= True, axis = 0)
#%%
print(df.isna().sum())
#Uklonjeni su uzorci za koje je falila vrednost prethodnih obelezja pri cemu je pojavljivanje nan vrednosti
#za ta obelezja manja od 1% stoga je bilo logicno ukloniti same redove tabela jer ne remeti bazu
#%%Za obelezja precipitation i Iprec za dopunu nan vrednosti cu koristiti dopunu vrednostima iz prethodnog sata
df['precipitation'].fillna(method='ffill', inplace = True)
df['Iprec'].fillna(method = 'ffill', inplace = True)
#%% Istrazivanjem na internetu dosla sam do zakljucka da je u tokom godina bilo raznih ekstremnih vrednosti 
#za PM2.5 cestice stoga nisu iznenadjujuci podaci koji se nalaze u datasetu
#nedostajajuce vrednosti cu popunjavati sa medijanom posto ima dosta nedostajajucih vrednosti
df['PM_US Post'].fillna(df['PM_US Post'].median(), inplace = True)

#%%Takodje obelezje No je samo redni broj obelezja stoga mozemo da izbacimo ovo obelezje
df.drop(['No'], inplace = True, axis = 1)
#%%
print(df.shape)
#%% Provera nelogicnih vrednosti 
checkForFalseValues = df.describe()
# Analizom na internetu dosla sam do zakljucka da su vrednosti koje su ostale u bazi
# podataka tacne
#%% Prikaz svih pravaca vetra 
print(df["cbwd"].unique())
#%% Zamena kategorickog obelezja cbwd sa numerickim vrednostima i to na sledeci nacin
df.replace(to_replace ="cv", value = 0, inplace = True)
df.replace(to_replace = "SE", value = 135, inplace = True)
df.replace(to_replace = "SW", value = 225, inplace = True)
df.replace(to_replace = "NW", value = 315, inplace = True)
df.replace(to_replace = "NE", value = 45, inplace = True)
#%%Nakon prethodnog koraka zavrsena je dopuna nedostajajucih podataka i izbacivanje laznih uzoraka
#%% --- Analiza obelezja ---
#Obelezje year predstavlja godinu u kojoj su mereni vremenski uslovi i to od 2010 do 2015
#Obelezje month uzima vrednosti od 1-12 i predstavlja mesec u kom su izmereni vremenski uslovi
#Obelezje day uzima vrednosti od 1-31
#Obelezje hour uzima vrednosti od 0-23
#Obelezje season predstavlja godisnje doba pri cemu je 4 - zima, 1 - prolece, 2 - leto, 3 - jesen
#Obelezje PM_US Post predstvalja koncentraciju PM2.5 cestica 
#Obelezje DEWP predstavlja temperaturu kondenzacije
#Obelezje HUMI predstavlja vlaznost vazduha
#Obelezje cbwd predstavlja pravac vetra
#Obelezje lws predstavlja kumulativnu brzinu vetra
#Obelezje precipatition predstavlja kolicinu padavina u toku svakog sata
#Obelezje Iprec predstavlja ukupne padavine u toku dana - sabira kolicinu padavina u toku dana do tog sata 
#koji predstavlja uzorak koji posmatramo
#%% Srednja vrednost i interkvartalni opseg numerickih obelezja koji ne predstavljaju kategorije
numericDesc = df.drop(['year','month','day','hour','season', 'cbwd'],axis = 1,inplace = False)
dfDsc = numericDesc.describe()
#%% Analiza temperature kroz godisnja doba

springWeather = df[df["season"] == 1]
summerWeather = df[df["season"] == 2]
fallWeather = df[df["season"] == 3]
winterWeather = df[df["season"] == 4]
#%% Analiza prolecnih temperatura
print("Prosecna prolecna temperatura u Sangaju : ", round(springWeather["TEMP"].mean(),2))
print("Maksimalna prolecna temperatura u Sangaju : ", round(springWeather["TEMP"].max(),2))
print("Minimalna prolecna temperatura u Sangaju : ", round(springWeather["TEMP"].min(),2))

#%% Analiza letnjih temperatura
print("Prosecna letnja temperatura u Sangaju : ", round(summerWeather["TEMP"].mean(),2))
print("Maksimalna letnja temperatura u Sangaju : ", round(summerWeather["TEMP"].max(),2))
print("Minimalna letnja temperatura u Sangaju : ", round(summerWeather["TEMP"].min(),2))
#%% Analiza jesenjih temperatura
print("Prosecna jesenja temperatura u Sangaju : ", round(fallWeather["TEMP"].mean(),2))
print("Maksimalna jesenja temperatura u Sangaju : ", round(fallWeather["TEMP"].max(),2))
print("Minimalna jesenja temperatura u Sangaju : ", round(fallWeather["TEMP"].min(),2))

#%% Analiza zimskih temperatura
print("Prosecna zimska temperatura u Sangaju : ", round(winterWeather["TEMP"].mean(),2))
print("Maksimalna zimska temperatura u Sangaju : ", round(winterWeather["TEMP"].max(),2))
print("Minimalna zimska temperatura u Sangaju : ", round(winterWeather["TEMP"].min(),2))

#%% Razdvajam temperature za prolece po godina i zatim ih prikazujem na grafiku kako bi uvideli razlike u godinama

springWeather2010 = springWeather[springWeather["year"] == 2010]
springWeather2011 = springWeather[springWeather["year"] == 2011]
springWeather2012 = springWeather[springWeather["year"] == 2012]
springWeather2013 = springWeather[springWeather["year"] == 2013]
springWeather2014 = springWeather[springWeather["year"] == 2014]
springWeather2015 = springWeather[springWeather["year"] == 2015]
#%%
gbySpringWeather2010 = springWeather2010.groupby(["year","month","day"])["TEMP"].mean().reset_index()
gbySpringWeather2011 = springWeather2011.groupby(["year","month","day"])["TEMP"].mean().reset_index()
gbySpringWeather2012 = springWeather2012.groupby(["year","month","day"])["TEMP"].mean().reset_index()
gbySpringWeather2013 = springWeather2013.groupby(["year","month","day"])["TEMP"].mean().reset_index()
gbySpringWeather2014 = springWeather2014.groupby(["year","month","day"])["TEMP"].mean().reset_index()
gbySpringWeather2015 = springWeather2015.groupby(["year","month","day"])["TEMP"].mean().reset_index()
#%% Spajanje u datum

gbySpringWeather2010["date"] =  gbySpringWeather2010["month"].astype(str) + "-" + gbySpringWeather2010["day"].astype(str)
gbySpringWeather2011["date"] =  gbySpringWeather2011["month"].astype(str) + "-" + gbySpringWeather2011["day"].astype(str)
gbySpringWeather2012["date"] =  gbySpringWeather2012["month"].astype(str) + "-" + gbySpringWeather2012["day"].astype(str)
gbySpringWeather2013["date"] =  gbySpringWeather2013["month"].astype(str) + "-" + gbySpringWeather2013["day"].astype(str)
gbySpringWeather2014["date"] =  gbySpringWeather2014["month"].astype(str) + "-" + gbySpringWeather2014["day"].astype(str)
gbySpringWeather2015["date"] =  gbySpringWeather2015["month"].astype(str) + "-" + gbySpringWeather2015["day"].astype(str)
#%% Iscrtavanje linijskog grafika za temperature

plt.figure()
plt.plot(gbySpringWeather2010["date"],gbySpringWeather2010["TEMP"],label="2010",color="purple")
plt.plot(gbySpringWeather2011["date"],gbySpringWeather2010["TEMP"],label="2011",color="black")
plt.plot(gbySpringWeather2012["date"],gbySpringWeather2010["TEMP"],label="2012",color="yellow")
plt.plot(gbySpringWeather2013["date"],gbySpringWeather2010["TEMP"],label="2013",color="blue")
plt.plot(gbySpringWeather2014["date"],gbySpringWeather2010["TEMP"],label="2014",color="red")
plt.plot(gbySpringWeather2015["date"],gbySpringWeather2010["TEMP"],label="2015",color="green")
plt.legend(loc="best")

#%% Analiza PM2.5 cestica

print("Prosecna koncentracija PM2.5 cestica u Sangaju : ", round(df["PM_US Post"].mean(),2))
print("Minimalna koncentracija PM2.5 cestica u Sangaju : ", round(df["PM_US Post"].min(),2))
print("Maksimalna koncentracija PM2.5 cestica u Sangaju : ", round(df["PM_US Post"].max(),2))
#%% Koncentracija PM2.5 cestica u toku godina

print("Prosecna koncentracija PM2.5 cestica u Sangaju 2010: ", round(df.groupby("year")["PM_US Post"].mean(),2))
print("Minimalna koncentracija PM2.5 cestica u Sangaju 2010: ", round(df.groupby("year")["PM_US Post"].min(),2))
print("Maksimalna koncentracija PM2.5 cestica u Sangaju 2010: ", round(df.groupby("year")["PM_US Post"].max(),2))
#%%
meanPM = df.groupby("year")["PM_US Post"].mean()
minPM = df.groupby("year")["PM_US Post"].min()
maxPM = df.groupby("year")["PM_US Post"].max()
#%% Graficki prikaz dobijenih rezultata za srednju,minimalnu i maksimalnu vrednost
# PM2.5 cestica u toku godina

plt.figure()
plt.plot(meanPM.index,meanPM.values)
plt.ylabel("Koncentracija PM2.5 cestica")
plt.xlabel("Godina")
plt.title("Prosecna koncentracija PM2.5 cestica")
#%%
plt.figure()
plt.plot(minPM.index,minPM.values)
plt.ylabel("Koncentracija PM2.5 cestica")
plt.xlabel("Godina")
plt.title("Minimalna koncentracija PM2.5 cestica")
#%%
plt.figure()
plt.plot(maxPM.index,maxPM.values)
plt.ylabel("Koncentracija PM2.5 cestica")
plt.xlabel("Godina")
plt.title("Maksimalna koncentracija PM2.5 cestica")
#%% Menjam kategoriju godisnjeg doba za tekst kako bi prikaz bio lepsi
dfSeasons = df
dfSeasons['season'] = dfSeasons['season'].astype('category')
dfSeasons['season'] = dfSeasons['season'].cat.rename_categories({1: 'Prolece', 2: 'Leto', 3: 'Jesen', 4: 'Zima'})

#%%
zima = dfSeasons[dfSeasons["season"] == "Zima"]
jesen = dfSeasons[dfSeasons["season"] == "Jesen"]
leto = dfSeasons[dfSeasons["season"] == "Leto"]
prolece = dfSeasons[dfSeasons["season"] == "Prolece"]
#%% Boxplot za kolicinu PM 2.5 cestica
plt.figure()
plt.boxplot([zima['PM_US Post'],prolece['PM_US Post'],leto['PM_US Post'], jesen['PM_US Post']]) 
plt.ylabel('Kolicina PM 2.5 cestica')
plt.xlabel('Godisnje doba')
plt.xticks([1, 2, 3, 4], ["Zima", "Prolece","Leto","Jesen"])
plt.grid()
#%% Zamena kategorickog obelezja cbwd sa kategorickim vrednostima i to na sledeci nacin
wind = df
wind.replace(to_replace = 0, value = "cv", inplace = True)
wind.replace(to_replace = 135, value = "SE", inplace = True)
wind.replace(to_replace = 225, value = "SW", inplace = True)
wind.replace(to_replace = 315, value = "NW", inplace = True)
wind.replace(to_replace = 45, value = "NE", inplace = True)
#%% Analiza koji je pravac vetra najcesci u Sangaju u toku svakog sata

windPositionGBY = wind.groupby(["cbwd"])["day"].count().reset_index()
#Najcesce duva severo-istocni vetar dok najredje duva jugo-zapadni
#%%Graficki prikaz
plt.figure()
plt.plot(windPositionGBY["cbwd"],windPositionGBY["day"])
plt.title("Najcesci pravac duvanja vetra")
plt.xlabel("Pravac duvanja vetra")
plt.ylabel("Broj sati")

#%% Analiza koji je pravac vetra najcesci po godisnjim dobima u toku svakog sata
windPositionSeasonsGBY = wind.groupby(["season","cbwd"])["hour"].count().reset_index()
#%%
windSpring = windPositionSeasonsGBY[windPositionSeasonsGBY["season"] == 'Prolece']
windSummer = windPositionSeasonsGBY[windPositionSeasonsGBY["season"] == 'Leto']
windFall = windPositionSeasonsGBY[windPositionSeasonsGBY["season"] == 'Jesen']
windWinter = windPositionSeasonsGBY[windPositionSeasonsGBY["season"] == 'Zima']

#%% Graficki prikaz

plt.figure()
plt.plot(windSpring["cbwd"],windSpring["hour"],color="green",label="Prolece")
plt.plot(windSummer["cbwd"],windSummer["hour"],color="yellow",label="Leto")
plt.plot(windFall["cbwd"],windFall["hour"],color="brown",label="Jesen")
plt.plot(windWinter["cbwd"],windWinter["hour"],color="blue",label="Zima")
plt.title("Pravac vetra po godisnjim dobima")
plt.xlabel("Pravac duvanja vetra")
plt.ylabel("Broj sati")
plt.legend(loc="best")

#%% Analiza vlaznosti vazduha u Sangaju

print("Prosecna vlaznost vazduha u Sangaju : ", round(df["HUMI"].mean(),2))
print("Minimalna vlaznost vazduha u Sangaju : ", round(df["HUMI"].min(),2))
print("Maksimalna vlaznost vazduha u Sangaju : ", round(df["HUMI"].max(),2))

#%% Analiza srednje vlaznosti vazduha u toku godisnjih doba

meanHumi = df.groupby("season")["HUMI"].mean().reset_index()
#%% Graficki prikaz

plt.figure()
plt.plot(meanHumi["season"],meanHumi["HUMI"])
plt.title("Prosecna vlaznost vazduha tokom godine")
plt.xlabel("Godisnje doba")
plt.ylabel("Vlaznost vazduha")
#Najveca vlaznost vazduha je u toku leta dok je najniza u toku proleca
#%% Analiza padavina u toku svakog dana 
#Prvo grupisemo podatke po godini

prec2010 = df[df["year"] == 2010]
prec2011 = df[df["year"] == 2011]
prec2012 = df[df["year"] == 2012]
prec2013 = df[df["year"] == 2013]
prec2014 = df[df["year"] == 2014]
prec2015 = df[df["year"] == 2015]
#%% Grupisemo podatke po mesecu i danu i racunamo ukupne dnevne padavine 

dailyPrec2010 = prec2010.groupby(["year","month","day"])["precipitation"].sum().reset_index()
dailyPrec2011 = prec2011.groupby(["year","month","day"])["precipitation"].sum().reset_index()
dailyPrec2012 = prec2012.groupby(["year","month","day"])["precipitation"].sum().reset_index()
dailyPrec2013 = prec2013.groupby(["year","month","day"])["precipitation"].sum().reset_index()
dailyPrec2014 = prec2014.groupby(["year","month","day"])["precipitation"].sum().reset_index()
dailyPrec2015 = prec2015.groupby(["year","month","day"])["precipitation"].sum().reset_index()

#%% Analiza koji dan je bio najkisovitiji u kojoj godini i koliko je taj dan palo kise

print("U toku 2010 palo je : " , dailyPrec2010["precipitation"].sum(), " mm kise, dok je u toku najkisovitijeg dana palo : ", round(dailyPrec2010["precipitation"].max(),2), "mm kise")
print("U toku 2011 palo je : " , dailyPrec2011["precipitation"].sum(), " mm kise, dok je u toku najkisovitijeg dana palo : ", round(dailyPrec2011["precipitation"].max(),2), "mm kise")
print("U toku 2012 palo je : " , dailyPrec2012["precipitation"].sum(), " mm kise, dok je u toku najkisovitijeg dana palo : ", round(dailyPrec2012["precipitation"].max(),2), "mm kise")
print("U toku 2013 palo je : " , dailyPrec2013["precipitation"].sum(), " mm kise, dok je u toku najkisovitijeg dana palo : ", round(dailyPrec2013["precipitation"].max(),2), "mm kise")
print("U toku 2014 palo je : " , dailyPrec2014["precipitation"].sum(), " mm kise, dok je u toku najkisovitijeg dana palo : ", round(dailyPrec2014["precipitation"].max(),2), "mm kise")
print("U toku 2015 palo je : " , dailyPrec2015["precipitation"].sum(), " mm kise, dok je u toku najkisovitijeg dana palo : ", round(dailyPrec2015["precipitation"].max(),2), "mm kise")

#Najkisovitija je bila 2012 godina dok je u toku jednog dana najvise palo kise takodje u toku 2012 godine
#Takodje mozemo uociti da postoje prilicne razlike u kolicini kise koja je pala u toku godina gde je u 
#toku 2011 palo samo 900 mm kise dok je naredne godine palo cak 400 mm vise 
#Takodje postoje i razlike u najvecim dnevnim padavinama gde je u toku 2014 najveca dnevna kolicina padavina
#iznosila 74.3 mm dok je u prethodne 2 godine palo preko 100 mm u toku jednog dana

#%% Analiza temperature vazduha u Sangaju

print("Najtoplijeg dana u 2010 godini izmerena temperatura je iznosila : ", round(prec2010["TEMP"].max(),2), ", dok je najniza temperatura bila : ", round(prec2010["TEMP"].min(),2), "a prosecna temperatura : ", round(prec2010["TEMP"].mean(),2))
print("Najtoplijeg dana u 2011 godini izmerena temperatura je iznosila : ", round(prec2011["TEMP"].max(),2), ", dok je najniza temperatura bila : ", round(prec2011["TEMP"].min(),2), "a prosecna temperatura : ", round(prec2011["TEMP"].mean(),2))
print("Najtoplijeg dana u 2012 godini izmerena temperatura je iznosila : ", round(prec2012["TEMP"].max(),2), ", dok je najniza temperatura bila : ", round(prec2012["TEMP"].min(),2), "a prosecna temperatura : ", round(prec2012["TEMP"].mean(),2))
print("Najtoplijeg dana u 2013 godini izmerena temperatura je iznosila : ", round(prec2013["TEMP"].max(),2), ", dok je najniza temperatura bila : ", round(prec2013["TEMP"].min(),2), "a prosecna temperatura : ", round(prec2013["TEMP"].mean(),2))
print("Najtoplijeg dana u 2014 godini izmerena temperatura je iznosila : ", round(prec2014["TEMP"].max(),2), ", dok je najniza temperatura bila : ", round(prec2014["TEMP"].min(),2), "a prosecna temperatura : ", round(prec2014["TEMP"].mean(),2))
print("Najtoplijeg dana u 2015 godini izmerena temperatura je iznosila : ", round(prec2015["TEMP"].max(),2), ", dok je najniza temperatura bila : ", round(prec2015["TEMP"].min(),2), "a prosecna temperatura : ", round(prec2015["TEMP"].mean(),2))

#Uocavamo da je prosecna temperatura tokom svih godina 17-18 stepeni sto nam govori da ipak nema 
#drasticnog globalnog zagrevanja u Sangaju, takodje najnize temperature su se kretale od -3 do -5 sto
#a najvise temperature od 37-41 stepen sto je pokazatelj da nije bilo prevelikih oscilacija u vremenskim
#uslovima sto se tice temperature vazduha
#%%Prosecan broj dana godisnje kada je koncentracija veca od nezdrave 35.4mg/m3
# grupisanje da bi se dobila prosecna dnevna kolicina PM 2.5 cestica
gbyPM25 = df.groupby(["year","month","day"])["PM_US Post"].mean().reset_index()
# filtriranje dana sa koncentracijiom vecom od 35.4
gbyPM25 = gbyPM25[gbyPM25["PM_US Post"] >= 35.4]
#%% Grupisanje po godinama kako bi dobili broj dana 
gbyPMyear = gbyPM25.groupby(["year"])["day"].count().reset_index()
#%%Poredjenje prosecne dnevne temperature sa normalnom raspodelom
gby = df.groupby(["year","month","day"])["TEMP"].mean().reset_index()
#%% Poredjenje temperature sa normalnom raspodelom
import seaborn as sb
from scipy.stats import norm
sb.distplot(gby["TEMP"], fit=norm)
plt.xlabel('Temperatura (℃)')
plt.ylabel('Verovatnoća')
plt.title("Poredjenje prosecne temperature sa normalnom raspodelom")
#Uocavamo da postoji znacajno odstupanje od normalne raspodele od maksimuma funkcije normalne raspodele
#dok repovi raspodele prosecne temperature ne odstupaju mnogo od funkcije normalne raspodele sto se tice
#oblika, zakljucujemo da je funkcija raspodele prosecne temperature spoljostenija od funkcije normalne
#raspodele
#%% Grupisanje da bismo dobili dnevne srednje vrednosti za sledeca obelezja
gbyDEWP = df.groupby(["year","month","day"])["DEWP"].mean().reset_index()
gbyHUMI = df.groupby(["year","month","day"])["HUMI"].mean().reset_index()
gbyPRES = df.groupby(["year","month","day"])["PRES"].mean().reset_index()
gbyIWS = df.groupby(["year","month","day"])["Iws"].mean().reset_index()
gbyPreciptation = df.groupby(["year","month","day"])["precipitation"].mean().reset_index()
gbyIprec = df.groupby(["year","month","day"])["Iprec"].mean().reset_index()
#%% Poredjenje ostalih numerickih obelezja sa normalnom raspodelom
import seaborn as sb
from scipy.stats import norm
fig, ax = plt.subplots(3, 2)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)
sb.distplot(gbyDEWP["DEWP"], fit=norm, ax = ax[0,0])
ax[0][0].set(xlabel="Temperatura kondenzacije", ylabel="")
sb.distplot(gbyHUMI["HUMI"], fit=norm, ax = ax[0,1])
ax[0][1].set(xlabel = "Vlaznost vazduha", ylabel="")
sb.distplot(gbyPRES["PRES"], fit=norm, ax = ax[1,0])
ax[1][0].set(xlabel = "Vazdusni pritisak", ylabel="")
sb.distplot(gbyIWS["Iws"], fit=norm, ax = ax[1,1])
ax[1][1].set(xlabel = "Kumulativna brzina vetra", ylabel="")
sb.distplot(gbyPreciptation["precipitation"], fit=norm, ax = ax[2,0])
ax[2][0].set(xlabel = "Padavine u toku sata", ylabel="")
sb.distplot(gbyIprec["Iprec"], fit=norm, ax = ax[2,1])
ax[2][1].set(xlabel = "Kumulativna kolicina padavina", ylabel="")
#%% Racunanje koeficijenta asimetrije i spljostenosti za prosecne dnevne temperature
from scipy.stats import kurtosis
from scipy.stats import skew

print('koef.asimetrije:  %.2f' % skew(gby['TEMP']))
print('koef.spljoštenosti:  %.2f' % kurtosis(gby['TEMP']))

#%% Analiza vazdusnog pritiska

print("Najvisi vazdusni pritisak koji je zabelezen u Sangaju je :", round(df["PRES"].max(),2), ", dok je najnizi zabelezen iznosio : ", round(df["PRES"].min(),2), "a srednja vrednost vazdusnog pritiska je :", round(df["PRES"].mean(),2))

#Znamo da je vazdusni pritisak povisen u toku toplijih dana dok je vazdusni pritisak smanjen
#u toku hladnijih dana - i stoga postoji izuzetna korelacija izmedju ova dva obelezja

#%% Analiza korelacije numerickih obelezja
onlyNumeric = df.drop(["year","month","day","hour","season"],axis = 1,inplace = False)
#%%
corr = onlyNumeric.corr()
f = plt.figure(figsize=(12, 9))
sb.heatmap(corr, annot=True);

#Uocavamo jaku negativnu korelaciju izmedju temperature kondenzacije i vazdusni pritisak
#Zatim jaku pozitivnu korelaciju izmedju temperature kondezacije i temperature
#Zatim jaku negativnu korelaciju izmedju vazdusnog pritiska i temperature
#Zatim izrazenu korelaciju izmedju temperature kondenzacije i vlaznosti vazduha
#%% Grupisanje po godini, mesecu i danu 
gbyBasic = df.groupby(["year","month","day"])["DEWP","PRES"].mean().reset_index()
#%% Iscrtavanje korelacije izmedju vazdusnog pritiska i temperature kondenzacije
plt.figure()
plt.scatter(gbyBasic["DEWP"],gbyBasic["PRES"])
plt.plot(np.unique(gbyBasic["DEWP"]), 
         np.poly1d(np.polyfit(gbyBasic["DEWP"], gbyBasic["PRES"], 1))
         (np.unique(gbyBasic["DEWP"])), color='red')
plt.xlabel("Temperatura kondenzacije")
plt.ylabel("Vazdusni pritisak")
#%% Grupisanje po godini, mesecu i danu
gbyCorrelation = df.groupby(["year","month","day"])["DEWP","PRES","TEMP","PM_US Post"].mean().reset_index()
#%% Iscrtavanje korelacije izmedju koncentracije PM 2.5 cestica i obelezja sa kojima ima najizrazeniju korelaciju
fig, ax = plt.subplots(3, 1)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)
ax[0].scatter(gbyCorrelation["DEWP"],gbyCorrelation["PM_US Post"])
plt.plot(np.unique(gbyCorrelation["DEWP"]), 
         np.poly1d(np.polyfit(gbyCorrelation["DEWP"], gbyCorrelation["PM_US Post"], 1))
         (np.unique(gbyCorrelation["DEWP"])), color='red')
ax[0].set(xlabel = "Temperatura kondenzacije")

ax[0].scatter(gbyCorrelation["DEWP"],gbyCorrelation["PM_US Post"])
ax[1].plot(np.unique(gbyCorrelation["PRES"]), 
         np.poly1d(np.polyfit(gbyCorrelation["PRES"], gbyCorrelation["PM_US Post"], 1))
         (np.unique(gbyCorrelation["PRES"])), color='red')
ax[1].set(xlabel = "Vazdusni pritisak")
ax[1].set(ylabel = "Kolicina PM 2.5 cestica")
ax[2].scatter(gbyCorrelation["TEMP"],gbyCorrelation["PM_US Post"])
plt.plot(np.unique(gbyCorrelation["TEMP"]), 
         np.poly1d(np.polyfit(gbyCorrelation["TEMP"], gbyCorrelation["PM_US Post"], 1))
         (np.unique(gbyCorrelation["TEMP"])), color='red')
ax[2].set(xlabel = "Temperatura")
#%%
gbyCorrelation.drop(["year","month","day"],axis = 1, inplace = True)
#%% pairplot korelacione matrice

sb.set()
sb.pairplot(gbyCorrelation, height = 2.5)
plt.show();
#%% boxplot za prosecne dnevne temperature
avgTemp = df.groupby(["year","month","day"])["TEMP"].mean().reset_index()
plt.boxplot(avgTemp["TEMP"]) 
plt.ylabel('Prosečna temperatura (℃)')
plt.xticks([1],["Sangaj"])
plt.grid()
#%% boxplot za prosecnu dnevnu izmerenu kolicinu PM2.5 cestica
avgPM = df.groupby(["year","month","day"])["PM_US Post"].mean().reset_index()
plt.boxplot(avgPM["PM_US Post"]) 
plt.ylabel('Prosečna kolicina PM 2.5 cestica')
plt.xticks([1],["Sangaj"])
plt.grid()

#%% Histogram temperatura u zavisnosti od godina kad su merene

plt.hist(prec2010["TEMP"], density=True, alpha=0.5, bins=50, label = '2010')
plt.hist(prec2011["TEMP"], density=True, alpha=0.5, bins=50, label = '2011')
plt.hist(prec2012["TEMP"], density=True, alpha=0.5, bins=50, label = '2012')
plt.hist(prec2013["TEMP"], density=True, alpha=0.5, bins=50, label = '2013')
plt.hist(prec2014["TEMP"], density=True, alpha=0.5, bins=50, label = '2014')
plt.hist(prec2015["TEMP"], density=True, alpha=0.5, bins=50, label = '2015')
plt.legend(loc = "best")
plt.xlabel("Temperatura")
plt.ylabel("Verovatnoca")
#%% Graficki prikaz zavisnosti numerickih obelezja
sb.set()
sb.pairplot(onlyNumeric, height = 2.5)
plt.show();
#%%  --- Analiza zavisnosti PM 2.5 obelezja od ostalih obelezja u bazi ---
#nijedno od obelezja nije u izrazenoj korelaciji sa PM2.5 cesticama 
#postoji slaba negativna korelacija izmedju PM2.5 cestica i temperature kondenzacije (-0.24)
#postoji veoma mala negativna korelacije izmedju PM2.5 cestica i vlaznosti vazduha (-0.066)
#postoji slaba pozitivna korelacija izmedju PM2.5 cestica i vazdusnog pritiska (0.21)
#postoji slaba negativna korelacija izmedju PM2.5 cestica i temperature vazduha (0.25)
#postoji slaba negativna korelacija izmedju PM2.5 cestica i brzine vetra (-0.14)
#postoji veoma slaba negativna korelacija izmedju PM2.5 cestica i kolicine padavina (-0.055)
#postoji veoma slaba negativna korelacija izmedju PM2.5 cestica i kumulativne kolicine padavina (-0.082)
#%% --- Linearna regresija ---
#%%Prvo odbacujemo sva kategoricka obelezja i obelezje kolicine PM 2.5 cestica iz skupa za predikciju
x = df.drop(["year","month","day","hour","season","cbwd","PM_US Post"], axis = 1)
y = df["PM_US Post"]
#%%
print(x.shape)
print(x.columns)
print(x.head())
#%% funkcija za merenje uspesnosti regresionog modela
def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) 
    mae = mean_absolute_error(y_test, y_predicted) 
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    print('Srednja kvadratna greska: ', mse)
    print('Srednja apsolutna greska: ', mae)
    print('Koren srednje kvadratne greske: ', rmse)
    print('R2: ', r2)
    print('R2 prilagodjen: ', r2_adj)

    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.title("Osnovni oblik linearne regresije")
print("koeficijenti: ", first_regression_model.coef_)
# Srednja apsolutna greska je 20.85, koren srednje kvadratne greske je 33.26 dok je R2 0.11
# iz cega mozemo zakljuciti da nas model pravi greske i predvidja vrednosti koje su bliske
# srednjoj vrednosti modela za obuku  
#%% Selekcija obelezja
import statsmodels.api as sm
X = sm.add_constant(x_train)

model = sm.OLS(y_train, X.astype('float')).fit()
#%%
print(model.summary())
# Zakljucujemo da treba da odbacimo obelezje DEWP i precipatation koji imaju vecu p vrednost od 0.01
#%% Odbacujemo obelezja
x_train.drop(["DEWP","precipitation"], axis = 1, inplace = True)
x_test.drop(["DEWP","precipitation"], axis = 1, inplace = True)
x.drop(["DEWP","precipitation"], axis = 1, inplace = True)
#%% Linearna regresija sa standardizovanim obelezjima
scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)
x_train_std.head()
#%% 
regression_model_std = LinearRegression()

# Obuka modela
regression_model_std.fit(x_train_std, y_train)

# Testiranje
y_predicted = regression_model_std.predict(x_test_std)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
plt.title("Linearna regresija sa standardizovanim obelezjima")
print("koeficijenti: ", regression_model_std.coef_)
# Srednja apsolutna greska je 20.85, koren srednje kvadratne greske je 33.26 dok je R2 0.11
# iz cega zakljucujemo da ne postoji napredak u odnosu na standardnu linearnu regresiju
#%% Linearna regresija sa hipotezom interakcije
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)

print(poly.get_feature_names())

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

# Inicijalizacija
regression_model_inter = LinearRegression()

# Obuka modela
regression_model_inter.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_inter.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
plt.title("Linearna regresija sa hipotezom interakcije")
print("koeficijenti: ", regression_model_inter.coef_)
# srednja apsolutna greska je 20.81, koren srednje kvadratne greske je 32.84 dok je R2 0.13 
# iz cega uocavamo da je doslo do poboljsanja rezultata regresije na test skupu u odnosu na
# prethodna 2 primenjena metoda
#%% Linearna regresija sa hipotezom interakciju i kvadrate
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...+d1x1^2+d2x2^2+...+dnxn^2

# Inicijalizacija
regression_model_degree = LinearRegression()

# Obuka modela
regression_model_degree.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_degree.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
plt.title("Linearna regresija sa hipotezom interakcije i kvadrata")
print("koeficijenti: ", regression_model_degree.coef_)
# srednja apsolutna greska je 20.82, koren srednje kvadratne greske je 32.71 dok je R2 0.14 
# iz cega uocavamo da je doslo do poboljsanja u odnsou na rezultate linearne regresije
# sa hipotezom interakcije
#%% Ridge regresija
# Inicijalizacija
ridge_model = Ridge(alpha=5)

# Obuka modela
ridge_model.fit(x_inter_train, y_train)

# Testiranje
y_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.title("Ridge regresija")
print("koeficijenti: ", ridge_model.coef_)
# srednja apsolutna greska je 20.82, koren srednje kvadratne greske je 32.71 dok je R2 0.14 iz cega
# zakljucujemo da daje priblizno iste rezultate kao i linearna regresija sa hipotezom interakcije i kvadrata
#%% Lasso regresija
# Model initialization
lasso_model = Lasso(alpha=0.0000001)

# Fit the data(train the model)
lasso_model.fit(x_inter_train, y_train)

# Predict
y_predicted = lasso_model.predict(x_inter_test)

# Evaluation
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


#ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.title("Lasso regresija")
print("koeficijenti: ", lasso_model.coef_)

# srednja apsolutna greska je 20.82, koren srednje kvadratne greske je 32.71 a R2 0.14 iz cega
# dolazimo do zakljucka da daje priblizno iste rezultate kao i Ridge regresija
