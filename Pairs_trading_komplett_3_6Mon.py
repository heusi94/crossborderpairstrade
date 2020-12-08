#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:23:04 2020

@author: jan
"""
#import of the necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import datetime
from statsmodels.regression.rolling import RollingOLS
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

#Preparation of the data
data_hdax = pd.read_csv(r"path_to_HDAX_data") #reads the CSV table with the data that was previously downloaded from Datastream
data_sp500 = pd.read_csv(r"path_to_SP500_data") #the same again for data of the S&P 500
dollar = pd.read_csv(r"path_to_exchangerate_data") #the same again for data of the Dollar-Euro exchange rate
dollar['Datum'] = pd.to_datetime(dollar['Datum']) #converts the column 'Date' into the format datetime 
data_hdax = data_hdax.rename({'RIC':'Name','Unnamed: 0':'Datum'}, axis='columns') #change of column captions
data_sp500 = data_sp500.rename({'RIC':'Name','Unnamed: 0':'Datum'}, axis='columns') #change of column captions

all_hdax = list(set(data_hdax['Name'])) #returns all tickers that appear at least once in the HDAX data set
all_sp500 = list(set(data_sp500['Name']))#returns all tickers that appear at least once in the S&P 500 data set


#writing the data into a dictionary
hdax_aktien = {} #creates empty dictionary
sp500_aktien ={} #creates empty dictionary
for ticker in all_hdax: #loop over all shares that appear at least once in the HDAX data set
    dat = pd.DataFrame(data_hdax.drop(data_hdax[data_hdax.Name!=ticker].index)) #drop all data with a name that is not equal to ticker 
    if len(dat['Datum'])>len(np.unique(dat['Datum'])): #sorts out the data that occur multiple times, if multiple, then only the first record is taken
       hdax_aktien[ticker] = dat[:len(np.unique(dat['Datum']))] #writes only the first complete dataset into dictionary
    else: #otherwise all data can be read in
        hdax_aktien[ticker] = dat #write data into dictionary

for ticker in all_sp500: #loop over all shares that appear at least once in the S&P 500 data set
    dat = pd.DataFrame(data_sp500.drop(data_sp500[data_sp500.Name!=ticker].index)) #drop all data with a name that is not equal to ticker
    if len(dat['Datum'])>len(np.unique(dat['Datum'])): #sorts out the data that occur multiple times, if multiple, then only the first record is taken
       sp500_aktien[ticker] = dat[:len(np.unique(dat['Datum']))]  #writes only the first complete dataset into dictionary
    else: #otherwise all data can be read in
        sp500_aktien[ticker] = dat #write data into dictionary


alle_daten = {**hdax_aktien, **sp500_aktien} #merges the data into one large dictionary

#Dividend adjustment: (DataStream unfortunately only provides split adjusted data. The dividend adjustment must be done manually)
all_members = all_hdax + all_sp500 #a new list with all tickers is created
alle_daten_adj = {} #an empty dictionary where the adjusted data is stored

for i in range (len(all_members)): #loop over all shares in all_members
   try: #start loop: if there is an error message, it does except --> Attention: Can't be stopped via stop button, only via kernel restart
        df = alle_daten[all_members[i]] #a time series is drawn
        df = df.sort_values(by=['Datum'], ascending=False) #this is sorted in descending order
        df = df.reset_index() #inserts "normal" index (1,2,3...)
        df['pre Adjusted Gross Dividend Amount'] = df['Adjusted Gross Dividend Amount'].shift(1) #Creates a new column where the dividend distribution is brought forward by one day. Is necessary to calculate the adjustment factor
        df['Faktor'] = np.where(np.isnan(df['pre Adjusted Gross Dividend Amount'])==True, 0 ,((df['CLOSE']-df['pre Adjusted Gross Dividend Amount'])/df['CLOSE'])) #calculates the factor
        positions = df.loc[pd.notna(df['pre Adjusted Gross Dividend Amount']), :].index #returns the indices (line number) of dividend distributions
        adj_faktor = df['Faktor'].iloc[positions].tolist() #writes the factors into a list
        positions = np.insert(positions, 0, 0.0) #a zero is inserted at the first position so that something happens before the first dividend payment
        positions = positions.append(pd.Index([len(df)])) #at the end the total number of lines is inserted, so that it is adjusted until the end
        cum_adj_faktor = np.cumprod(adj_faktor).tolist() #cumprod of factors is created in a list
        cum_adj_faktor = np.insert(cum_adj_faktor, 0, 1.0) #at the first position a 1 is inserted, since no adjustment is necessary at the beginning
        df['adjustierung'] = 0 #new column Adjustment is created and filled with 0
        for j in range(len(cum_adj_faktor)): #here the actual adjustment is made
            df['adjustierung'].loc[positions[j]:positions[j+1]-1] = df['CLOSE']*cum_adj_faktor[j] 
            #the dataframe is subdivided by the positions each part is multiplied by a different factor
        alle_daten_adj[all_members[i]] = df #the so created DataFrame is inserted into the dictionary
   except:
        print(i, 121212) #if try produces an error message, error code 121212 is output

#Insert #log-prices, because wethis will calculate the cointegration with this
for i in range (len(all_members)): #loop over all shares in the dictionary
   try:
       alle_daten_adj[all_members[i]]['Log-Preis adjustiert'] = np.log(alle_daten_adj[all_members[i]]['adjustierung']) #for each share in the dictionary a new column with the log price is inserted
   except:
        print(i, 131313) #if try produces an error message, the error code is output

#Conversion of the HDAX share price into USD
for i in range (len(members_hdax)): #loop over all shares in HDAX
   try: 
       alle_daten_adj[members_hdax[i]] = pd.merge(alle_daten_adj[members_hdax[i]], dollar[['Schlusskurs', 'Datum']], on='Datum' ) #the dollar rate is pinned as a new column
       alle_daten_adj[members_hdax[i]]['CLOSE_USD'] = alle_daten_adj[members_hdax[i]]['adjustierung']*alle_daten_adj[members_hdax[i]]['Schlusskurs'] #a new column with the closing price in USD is calculated
       alle_daten_adj[members_hdax[i]]['Log-Preis adjustiert'] = np.log(alle_daten_adj[members_hdax[i]]['CLOSE_USD']) #a new column is created with the log price of the exchange rate in USD
   except:
        print(i, 141414) #if try produces an error message, the error code is output
 

#Calculation of the cointegration values
hdax_index = pd.read_excel(r"path_to_HDAX_members", index_col=[0]) #reads the list of Excel with the monthly HDAX members, which was previously created from DataStream data
sp500_index = pd.read_excel(r"path_to_S&P500_members", index_col=[0]) #reads the list of Excel with the monthly S&P 500 members, which was previously created from DataStream data

datum = hdax_index.index.values.tolist() #creates a list of all times (usually the 30th of the month) when the members of the index are available

list = [] #creates an empty list in which the results are collected before they are transferred to the DataFrame
zwspeicher = pd.DataFrame(columns = ['Aktie1', 'Aktie2', 'Datum', 'Wert', 'Wert_gedreht'])  #creates an empty DataFrame (table) for the results

#Loop to calculatte the cointegration values
for k in range(0,len(datum)): #loop over the points in time
    date = datetime.date.fromtimestamp(datum[k]/1000000000) #converts the kth entry of the Date list into a suitable date format. The 1000... is necessary, because the time is stored in date with milliseconds accuracy.
    a = hdax_index.at[date, 'Index Constituents'] #reads the members at date into the variable, at = command for localization
    a = a.replace("'","") #deletes unnecessary things
    a = a.replace("[","")
    a = a.replace("]","")
    a = a.replace(" ","")
    members_hdax = a.split(",") #separates the variable a (in this case the members) at each comma into a list element
    
    b = sp500_index.at[date, 'Index Constituents'] #reads the members at date into the variable, at = command for localization
    b = b.replace("'","") #deletes unnecessary things
    b = b.replace("[","")
    b = b.replace("]","")
    b = b.replace(" ","")
    members_sp500 = b.split(",") #separates the variable a (in this case the members) at each comma into a list element

    for i in range(len(members_hdax)): #loop over the first share len(members_hdax)
        try:
            daten_3 = alle_daten_adj[members_hdax[i]] #reads the data for the HDAX stock
            daten_3 = daten_3.copy()
            daten_3['Datum'] = pd.to_datetime(daten_3['Datum']) #convert columns Datum to datetime format
            for j in range(len(members_sp500)): #len(members_sp500)
                    try:
                        daten_4 = alle_daten_adj[members_sp500[j]] #reads the data for the HDAX stock
                        daten_4 = daten_4.copy()
                        daten_4['Datum'] = pd.to_datetime(daten_4['Datum']) #convert columns Datum to datetime format
                        data = pd.merge(daten_3, daten_4, on='Datum') #the two time series are combined in one dataframe
                        data = data.sort_values(by='Datum', ascending=True) #sorts the list by date in ascending order
                        data['Index'] = data.index #the index is inserted as a separate column in the dataframe
                        monatsanfang = data.groupby(pd.DatetimeIndex(data.Datum).to_period('M')).nth(0) #searches the first trading day of every month
                        start = monatsanfang.loc[monatsanfang['Datum'] > pd.Timestamp(date)][:1]['Index'] #indicates the last trading day before the first trading day of the current month
                        data = data.sort_values(by='Datum', ascending=False) #sorts the list by date in descending order
                        jahresdaten = data[start[0]+1:start[0]+253] #extracts the price data of the last 252 trading days starting from start
                        jahresdaten = jahresdaten.set_index('Datum') #change index of Jahresdaten to date
                        jahresdaten = jahresdaten.sort_values(by='Datum', ascending=True) #sorts jahresdaten ascending
                        if adfuller(jahresdaten['Log-Preis adjustiert_y'])[1]>0.05 and adfuller(jahresdaten['Log-Preis adjustiert_USD'])[1]>0.05: #test for non-stationarity (p>0.05 = not stationary) --> both time series should be non-stationary
                            S1 =  sm.add_constant(jahresdaten['Log-Preis adjustiert_y']) #add a constant for the regressions
                            S2 = jahresdaten['Log-Preis adjustiert_USD']
                            ols_result1 = sm.OLS(S2, S1).fit() #regression of the two time-series
                            erg = ts.adfuller(ols_result1.resid)[1] #Dickey-Fuller test on the spread of the two timeseries
                            S1 = jahresdaten['Log-Preis adjustiert_y']
                            S2 = sm.add_constant(jahresdaten['Log-Preis adjustiert_USD']) #add a constant for the regressions
                            ols_result2 = sm.OLS(S1, S2).fit() #regression of the two time-series
                            erg_gedreht = ts.adfuller(ols_result2.resid)[1] #Dickey-Fuller test on the spread of the two timeseries
                            list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':erg, 'Wert_gedreht':erg_gedreht}) #write results into DataFrame
                            print(k, i, j, erg, erg_gedreht) 
                        elif adfuller(jahresdaten['Log-Preis adjustiert_y'])[1]<0.05 and adfuller(jahresdatenv)[1]<0.05: #if both time-series are stationary do this
                            list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':-1, 'Wert_gedreht':-1}) #write the pair with value -1 into DataFrame
                            print(k, i,j, -1)
                        else: #if one stock is stationary and the other is not, do this
                            list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':999999, 'Wert_gedreht':999999}) #write the pair with value 999999 into DataFrame
                            print(k, i,j, 999999)
                    except: #if there is any error during the process do this
                        list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':111111, 'Wert_gedreht':111111}) #write the pair with value 111111 into DataFrame
                        print(k, i,j, 111111)
        except: #if there is any error during the process do this
            list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':222222, 'Wert_gedreht':222222}) #write the pair with value 222222 into DataFrame
            print(k, i,j, 222222)

#Finding qualified pairs            
zwspeicher = zwspeicher.append(list) #transfer list to Dataframe
jahresauswahl = zwspeicher
qualpaare = jahresauswahl[(jahresauswahl['Wert'] < 0.05) & (jahresauswahl['Wert_gedreht'] < 0.05)] #only pairs with a p-value smaller than 0.05 in both directions are considered
#Reducing the qualified pairs (only until the end of 2018, so that they can be traded for another year)
qualpaare = qualpaare[alle_qualpaare['Datum'] >= datetime.date(2012,8,30) ]
qualpaare = qualpaare[qualpaare['Datum'] < datetime.date(2019,7,30)] #datetime.date(2018,12,31)

#open list to store results:
ergebnisse1 = []
daten = []
anzahl_nicht_geöffnet=0
anzahl_geöffnet=0
short_trades_liste = []
long_trades_liste = []
einstiegstag_liste = []
ausstiegstag_liste = []
performance_leg1 = []
performance_leg2 = []
positionmean = []
tradessum = []

#define entry and exit values
entry = 2
exit_schwelle = 0
stop_loss = 3.5


#calculation of the performance
for q in range(len(qualpaare)): #loop over all qualified pairs
        stock1 = alle_daten_adj[qualpaare.iloc[q,0]] #getting the data for the first stock of the pair
        stock2 = alle_daten_adj[qualpaare.iloc[q,1]] #getting the data for the second stock of the pair
        stock1 = stock1.copy()
        stock1['Datum'] = pd.to_datetime(stock1.Datum) #change format of column Datum
        stock2 = stock2.copy()
        stock2['Datum'] = pd.to_datetime(stock2.Datum) #change format of column Datum
        datum3 = qualpaare.iloc[q,2] #get the date of the qualification of the pair
        data2 = pd.merge(stock1, stock2, on='Datum') #unit the two DataFrames into one
        data2['Index'] = data2.index #set new index
        data2 = data2.sort_values(by='Datum', ascending=True) #sort the DataFrame ascending 
        monatsanfang = data2.groupby(pd.DatetimeIndex(data2.Datum).to_period('M')).nth(0) #get the first trading day of the every month
        start = monatsanfang.loc[monatsanfang['Datum'] > pd.Timestamp(datum3)].head(1)['Index'] #get the next first trading day from the day of qualification
        global jahresdaten2 #sets the variable as global
        data2 = data2.sort_values(by='Datum', ascending=False) #sort the DataFrame descending
        jahresdaten2 = data2[start[0]-126:start[0]+505][['Datum', 'adjustierung_x', 'adjustierung_y', 'Schlusskurs']] #zieht sich die Daten der letzten zwei Jahre und des kommenden Jahres
        jahresdaten2['adjustierung_x'] = jahresdaten2['adjustierung_x']*jahresdaten2['Schlusskurs']        
        jahresdaten2 = jahresdaten2.sort_values(by='Datum', ascending=True)
        jahresdaten2.index = jahresdaten2['Datum'] #das Datum wird als Index gesetzt
        y = sm.add_constant(jahresdaten2['adjustierung_y'])
        x = jahresdaten2['adjustierung_x']
        model = RollingOLS(x,y, window=252).fit() #Rollierende Regression über die letzten 252-Handelstage
        jahresdaten2['OLS'] = model.params['adjustierung_y'] #
        jahresdaten2['Erw'] = abs(jahresdaten2['adjustierung_y'] * jahresdaten2['OLS']) #der Erwartungswert für den Close-Price von x wird berechnet
        jahresdaten2['Spread'] = jahresdaten2['adjustierung_x'] - jahresdaten2['Erw'] #die Differenz zwischen tatsächlichem CLose und erwartetem wird berechnet
        jahresdaten2['Std'] = jahresdaten2['Spread'].rolling(252).std() #vom Spread wird die Standardabweichung rollierend über die letzten 252 Handelstage berechnet
        jahresdaten2['Mean'] = jahresdaten2['Spread'].rolling(252).mean() #und der Durchschnitt des Spreads über die letzten 252 Handelstag
        jahresdaten2['Zscore'] = (jahresdaten2['Spread']-jahresdaten2['Mean'])/jahresdaten2['Std'] #aus diesen Daten wird der Z-Score für jeden Tag berechnet
        jahresdaten2['Zscore_shift'] = jahresdaten2['Zscore'].shift(1)        
        #jahresdaten2 = jahresdaten2.dropna() #alle Daten mit fehlenden Werten (durch zweimal rollierend 252 idR 504 ersten Werte) werden geläscht
        jahresdaten2 = jahresdaten2[jahresdaten2['Datum'] > pd.Timestamp(datum3)]
        jahresdaten2 = jahresdaten2.copy()
        jahresdaten2['long'] = np.where((jahresdaten2['Zscore_shift'] < -entry) & (jahresdaten2['Zscore'] >= -entry),1,0)
        jahresdaten2['short'] = np.where((jahresdaten2['Zscore_shift'] > entry) & (jahresdaten2['Zscore'] <= entry),1,0)
        jahresdaten2['exit'] = (((jahresdaten2['Zscore']<=exit_schwelle) & (jahresdaten2['Zscore_shift']>exit_schwelle)) | ((jahresdaten2['Zscore']>=exit_schwelle) & (jahresdaten2['Zscore_shift']<exit_schwelle)) | ((jahresdaten2['Zscore']>=5) & (jahresdaten2['Zscore_shift']<5)) | ((jahresdaten2['Zscore']<=-5) & (jahresdaten2['Zscore_shift']>-5)) )*1.0
        a = len(jahresdaten2)-20
        jahresdaten2.iloc[a:,[11]] = 0.0   #nach 51-31.=20. Handeltag werden keine neuen Signale mehr generiert --> max. Haltedauer: 30 Tage
        jahresdaten2.iloc[a:,[12]] = 0.0  #Shortseite
        jahresdaten2.iloc[-1:,[13]] = 1.0 #Exit-Signal am letzten 51. Handelstag
        #Erstellen der Spalten, in welche anschließend unsere aktuelle Positionierung geschrieben wird
        jahresdaten2['long_market'] = 0.0
        jahresdaten2['short_market'] = 0.0
        #diese Variablen tracken die Marktposition während der Iteration
        long_market = 0
        short_market = 0
        #hier wird die Position an jedem einzelnen Tag geprüft
        for i, b in enumerate(jahresdaten2.iterrows()):
            if b[1]['long'] == 1.0:
                long_market = 1
            if b[1]['short'] == 1.0:
                short_market = 1
            if b[1]['exit'] == 1.0:
                long_market = 0
                short_market = 0  
            jahresdaten2.iloc[i, 14] = long_market
            jahresdaten2.iloc[i, 15] = short_market
        jahresdaten2 = jahresdaten2.reset_index(drop=True)
        jahresdaten2['long_market_shift'] = jahresdaten2['long_market'].shift()
        jahresdaten2 = pd.merge(jahresdaten2, dollar[{'Datum', 'Schlusskurs'}], on='Datum')
        jahresdaten3 = jahresdaten2.copy()
        sym1 = qualpaare.iloc[i,0]
        sym2 = qualpaare.iloc[i,1]
        jahresdaten3.set_index('Datum', drop=True, inplace=True)
        global portfolio
        portfolio = pd.DataFrame(index=jahresdaten3.index)
        portfolio['positions'] = jahresdaten3['long_market'] - jahresdaten3['short_market'] #gibt die Position an
        portfolio[sym1] = jahresdaten3['adjustierung_x'] #gibt die Kapitalentwicklung der ersten Position an
        portfolio[sym2] = jahresdaten3['adjustierung_y'] #gibt die Kapitalentwicklung der zweiten Position an
        #portfolio['beta'] = jahresdaten3['OLS']
        #portfolio['total'] = portfolio[sym1] + portfolio[sym2]
        portfolio['sym1returns'] = portfolio[sym1].pct_change()* portfolio['positions'].shift() #Performance der ersten Position
        portfolio['sym2returns'] = portfolio[sym2].pct_change()* -portfolio['positions'].shift() #Performance der zweiten 
        #portfolio['ER'] = jahresdaten3['Schlusskurs']
        #portfolio['ER_returns'] = jahresdaten3['Schlusskurs'].pct_change()*portfolio['positions'].shift()
        portfolio['posisions_value_x'] = (portfolio['sym1returns']+1).cumprod()
        portfolio['posisions_value_y'] = (portfolio['sym2returns']+1).cumprod()
        #portfolio['ER_Value'] = (portfolio['ER_returns']+1).cumprod()
        performance = ((portfolio['posisions_value_x'][-1:]-1)+(portfolio['posisions_value_y'][-1:]-1)).values
        datum = portfolio.index[-1].date()
        ergebnisse1.append(performance[0])
        daten.append(datum)
        jahresdaten3['long_market_shift2'] = jahresdaten3['long_market'].shift()
        jahresdaten3['short_market_shift2'] = jahresdaten3['short_market'].shift()
        jahresdaten3['long_trades_zähler'] = np.where((jahresdaten3['long_market_shift2']==0) & (jahresdaten3['long_market']==1)|(pd.isna(jahresdaten3['long_market_shift2']) & (jahresdaten3['long_market']==1)),1,0)
        jahresdaten3['short_trades_zähler'] = np.where((jahresdaten3['short_market_shift2']==0) & (jahresdaten3['short_market']==1)|(pd.isna(jahresdaten3['short_market_shift2']) & (jahresdaten3['short_market']==1)),1,0)
        jahresdaten3['trades_zähler'] = jahresdaten3['long_trades_zähler'] + jahresdaten3['short_trades_zähler']
        short_trades_liste.append(jahresdaten3['short_trades_zähler'].sum())
        long_trades_liste.append(jahresdaten3['long_trades_zähler'].sum())
        einstiegstag_liste.append((np.where((jahresdaten3['short_trades_zähler']==1) | (jahresdaten3['long_trades_zähler']==1)))[0])
        ausstiegstag_liste.append(np.where(((jahresdaten3['long_market']==0) & (jahresdaten3['long_market_shift']==1)) | ((jahresdaten3['short_market']==0) & (jahresdaten3['short_market_shift2']==1)))[0])
        positionmean.append(abs(portfolio['positions']).mean())
        tradessum.append(jahresdaten3['trades_zähler'].sum())
        performance_leg1.append(portfolio['posisions_value_x'][-1]-1)
        performance_leg2.append(portfolio['posisions_value_y'][-1]-1)
        print(q)
#    except:
   #     print(q,  "Fehler")
print(start, start2, start3, start4, start5)

Auswertung = pd.DataFrame({'Performance': ergebnisse1, 
                           'Datum': daten,
                           'Anzahl Short Trades': short_trades_liste,
                           'Anzahl Long Trades': long_trades_liste,
                           'Einstiegstage': einstiegstag_liste,
                           'Ausstiegstage': ausstiegstag_liste,
                           'Performance Leg1': performance_leg1,
                           'Performance Leg2': performance_leg2,                           
                           })





for q in range(len(qualpaare)):   #len(qualpaare)
    #try:
        stock1 = alle_daten_adj[qualpaare.iloc[q,0]] #zieht sich zuerst die Daten des Paares
        stock2 = alle_daten_adj[qualpaare.iloc[q,1]]
        stock1 = stock1.copy()
        stock1['Datum'] = pd.to_datetime(stock1.Datum)
        stock2 = stock2.copy()
        stock2['Datum'] = pd.to_datetime(stock2.Datum)
        datum3 = qualpaare.iloc[q,2] #und das Datum der Qualifikation als Paar
        data2 = pd.merge(stock1, stock2, on='Datum') #die zwei Zeitreihen werden in einem Dataframe vereint
        data2['Index'] = data2.index #der Index wird als eigene Spalte in den Dataframe eingefügt
        data2 = data2.sort_values(by='Datum', ascending=True) #sortiert die Liste nach Datum (dürfte absteigend sein)
        monatsanfang = data2.groupby(pd.DatetimeIndex(data2.Datum).to_period('M')).nth(0) #sucht jeweils den ersten Handelstag im Monat
        start = monatsanfang.loc[monatsanfang['Datum'] > pd.Timestamp(datum3)].head(1)['Index'] #gibt den nächsten ersten Handelstag ausgehend von date aus
        global jahresdaten2 #muss gesetzt werden, damit anschließend auch außerhalb der Funktion auf die Variable zugegriffen werden kann
        #hier nochmal schauen, ob es auch die richtigen Zeitpunkte zieht. Kann sein, dass die Daten nun anders sortiert sind es deshalb nicht mehr passt
        data2 = data2.sort_values(by='Datum', ascending=False)
        jahresdaten2 = data2[start[0]-126:start[0]+505][['Datum', 'adjustierung_x', 'adjustierung_y', 'Schlusskurs']] #zieht sich die Daten der letzten zwei Jahre und des kommenden Jahres
        jahresdaten2['adjustierung_x'] = jahresdaten2['adjustierung_x']*jahresdaten2['Schlusskurs']
        jahresdaten2 = jahresdaten2.sort_values(by='Datum', ascending=True)
        jahresdaten2.index = jahresdaten2['Datum'] #das Datum wird als Index gesetzt
        model = RollingOLS(jahresdaten2['adjustierung_x'],jahresdaten2['adjustierung_y'], window=252).fit() #Rollierende Regression über die letzten 252-Handelstage
        jahresdaten2['OLS'] = model.params #Der Regressions-Parameter wird als eigene Spalte eingesetzt
        jahresdaten2['Erw'] = jahresdaten2['adjustierung_y'] * jahresdaten2['OLS'] #der Erwartungswert für den Close-Price von x wird berechnet
        jahresdaten2['Spread'] = jahresdaten2['adjustierung_y'] - jahresdaten2['Erw'] #die Differenz zwischen tatsächlichem CLose und erwartetem wird berechnet
        jahresdaten2['Std'] = jahresdaten2['Spread'].rolling(252).std() #vom Spread wird die Standardabweichung rollierend über die letzten 252 Handelstage berechnet
        jahresdaten2['Mean'] = jahresdaten2['Spread'].rolling(252).mean() #und der Durchschnitt des Spreads über die letzten 252 Handelstag
        jahresdaten2['Zscore'] = (jahresdaten2['Spread']-jahresdaten2['Mean'])/jahresdaten2['Std'] #aus diesen Daten wird der Z-Score für jeden Tag berechnet
        jahresdaten2['Zscore_shift'] = jahresdaten2['Zscore'].shift(1)        
        #jahresdaten2 = jahresdaten2.dropna() #alle Daten mit fehlenden Werten (durch zweimal rollierend 252 idR 504 ersten Werte) werden geläscht
        jahresdaten2 = jahresdaten2[jahresdaten2['Datum'] > pd.Timestamp(datum3)]
        jahresdaten2 = jahresdaten2.copy()
        jahresdaten2['long'] = np.where((jahresdaten2['Zscore_shift'] < -entry) & (jahresdaten2['Zscore'] >= -entry),1,0)
        jahresdaten2['short'] = np.where((jahresdaten2['Zscore_shift'] > entry) & (jahresdaten2['Zscore'] <= entry),1,0)
        jahresdaten2['exit'] = (((jahresdaten2['Zscore']<=exit_schwelle) & (jahresdaten2['Zscore_shift']>exit_schwelle)) | ((jahresdaten2['Zscore']>=exit_schwelle) & (jahresdaten2['Zscore_shift']<exit_schwelle)) | ((jahresdaten2['Zscore']>=3.5) & (jahresdaten2['Zscore_shift']<3.5)) | ((jahresdaten2['Zscore']<=-3.5) & (jahresdaten2['Zscore_shift']>3.5)) )*1.0
        #jahresdaten2.iloc[-31:,[10]] = 0.0   #nach 51-31.=20. Handeltag werden keine neuen Signale mehr generiert --> max. Haltedauer: 30 Tage
        #jahresdaten2.iloc[-31:,[11]] = 0.0  #Shortseite
        jahresdaten2.iloc[-1:,[12]] = 1.0 #Exit-Signal am letzten 51. Handelstag
        #Erstellen der Spalten, in welche anschließend unsere aktuelle Positionierung geschrieben wird
        jahresdaten2['long_market'] = 0.0
        jahresdaten2['short_market'] = 0.0
        #diese Variablen tracken die Marktposition während der Iteration
        long_market = 0
        short_market = 0
        #hier wird die Position an jedem einzelnen Tag geprüft
        for i, b in enumerate(jahresdaten2.iterrows()):
            if b[1]['long'] == 1.0:
                long_market = 1
            if b[1]['short'] == 1.0:
                short_market = 1
            if b[1]['exit'] == 1.0:
                long_market = 0
                short_market = 0  
            jahresdaten2.iloc[i, 13] = long_market
            jahresdaten2.iloc[i, 14] = short_market
        jahresdaten2 = jahresdaten2.reset_index(drop=True)
        jahresdaten2['long_market_shift'] = jahresdaten2['long_market'].shift()
        jahresdaten2 = pd.merge(jahresdaten2, dollar[{'Datum', 'Schlusskurs'}], on='Datum')
        jahresdaten3 = jahresdaten2.copy()
        sym1 = qualpaare.iloc[i,0]
        sym2 = qualpaare.iloc[i,1]
        jahresdaten3.set_index('Datum', drop=True, inplace=True)
        global portfolio
        portfolio = pd.DataFrame(index=jahresdaten3.index)
        portfolio['positions'] = jahresdaten3['long_market'] - jahresdaten3['short_market'] #gibt die Position an
        portfolio[sym1] = jahresdaten3['adjustierung_x'] #gibt die Kapitalentwicklung der ersten Position an
        portfolio[sym2] = jahresdaten3['adjustierung_y'] #gibt die Kapitalentwicklung der zweiten Position an
        #portfolio['beta'] = jahresdaten3['OLS']
        #portfolio['total'] = portfolio[sym1] + portfolio[sym2]
        portfolio['sym1returns'] = portfolio[sym1].pct_change()* portfolio['positions'].shift() #Performance der ersten Position
        portfolio['sym2returns'] = portfolio[sym2].pct_change()* -portfolio['positions'].shift() #Performance der zweiten 
        #portfolio['ER'] = jahresdaten3['Schlusskurs']
        #portfolio['ER_returns'] = jahresdaten3['Schlusskurs'].pct_change()*portfolio['positions'].shift()
        portfolio['posisions_value_x'] = (portfolio['sym1returns']+1).cumprod()
        portfolio['posisions_value_y'] = (portfolio['sym2returns']+1).cumprod()
        #portfolio['ER_Value'] = (portfolio['ER_returns']+1).cumprod()
        performance = ((portfolio['posisions_value_x'][-1:]-1)+(portfolio['posisions_value_y'][-1:]-1)).values
        datum = portfolio.index[-1].date()
        ergebnisse1.append(performance[0])
        daten.append(datum)
        jahresdaten3['long_market_shift2'] = jahresdaten3['long_market'].shift()
        jahresdaten3['short_market_shift2'] = jahresdaten3['short_market'].shift()
        jahresdaten3['long_trades_zähler'] = np.where((jahresdaten3['long_market_shift2']==0) & (jahresdaten3['long_market']==1)|(pd.isna(jahresdaten3['long_market_shift2']) & (jahresdaten3['long_market']==1)),1,0)
        jahresdaten3['short_trades_zähler'] = np.where((jahresdaten3['short_market_shift2']==0) & (jahresdaten3['short_market']==1)|(pd.isna(jahresdaten3['short_market_shift2']) & (jahresdaten3['short_market']==1)),1,0)
        short_trades_liste.append(jahresdaten3['short_trades_zähler'].sum())
        long_trades_liste.append(jahresdaten3['long_trades_zähler'].sum())
        einstiegstag_liste.append((np.where((jahresdaten3['short_trades_zähler']==1) | (jahresdaten3['long_trades_zähler']==1)))[0])
        print(q)
    #except:
     #   print(q,  "Fehler")



qualpaare.iloc[93]
ergebnisse1[93]


#Backtest: https://www.quantstart.com/articles/Backtesting-An-Intraday-Mean-Reversion-Pairs-Strategy-Between-SPY-And-IWM/

ergebnisse1 = [] #eine leere Liste für die Performance jedes einzelnen Trades wird erstellt
daten = [] #eine leere Liste für das Enddatum jedes einzelnen Trades wird erstellt
short_trades_liste = [] #eine leere Liste für den Trade-Zähler wird erstellt
long_trades_liste = [] #eine leere Liste für den Trade-Zähler wird erstellt
einstiegstag_liste = [] #eine leere Liste für das Einstiegsdatum jedes einzelnen Trades wird erstellt

entry=2 #definiert die Einsteigsschwelle des Z-Scores
exit_schwelle=0  #definiert die Ausstiegsschwelle des Z-Scores 
anzahl_nicht_geöffnet=0 #setzt die Startwerte für die Zähler nicht-geöffneter Paare auf 0
anzahl_geöffnet=0 #setzt die Startwerte für die Zähler geöffneter Paare auf 0

#Loop über die qualifizierten Paare und Berechnung der Performance
for q in range(len(qualpaare)):   
    try:
        stock1 = alle_daten_adj[qualpaare.iloc[q,0]] #zieht sich zuerst die kurs-Daten des Paares
        stock2 = alle_daten_adj[qualpaare.iloc[q,1]] #zieht sich zuerst die kurs-Daten des Paares
        stock1 = stock1.copy() #copy-Befehl ist hier notwendig, damit an dieser Kopie anschließend Veränderungen vorgenommen werden können
        stock1['Datum'] = pd.to_datetime(stock1.Datum) #formatiert die Spalte Datum ins Datumsformat
        stock2 = stock2.copy() #copy-Befehl ist hier notwendig, damit an dieser Kopie anschließend Veränderungen vorgenommen werden können
        stock2['Datum'] = pd.to_datetime(stock2.Datum) #formatiert die Spalte Datum ins Datumsformat
        datum3 = qualpaare.iloc[q,2] #zieht das Datum, zu der das Paar qualifiziert wurde
        data2 = pd.merge(stock1, stock2, on='Datum') #die zwei Zeitreihen werden in einem Dataframe vereint
        data2['Index'] = data2.index #der Index wird als eigene Spalte in den Dataframe eingefügt
        data2 = data2.sort_values(by='Datum', ascending=True) #sortiert die Liste nach Datum
        monatsanfang = data2.groupby(pd.DatetimeIndex(data2.Datum).to_period('M')).nth(0) #sucht jeweils den ersten Handelstag im Monat
        start = monatsanfang.loc[monatsanfang['Datum'] > pd.Timestamp(datum3)].head(1)['Index'] #gibt den nächsten ersten Handelstag ausgehend von date aus
        global jahresdaten2 #muss gesetzt werden, damit anschließend auch außerhalb der Funktion auf die Variable zugegriffen werden kann
        data2 = data2.sort_values(by='Datum', ascending=False) #erneutes sortieren der Liste
        jahresdaten2 = data2[start[0]-126:start[0]+505][['Datum', 'adjustierung_x', 'adjustierung_y']] #zieht sich die Daten der letzten zwei Jahre und der folgenden sechs Monate
        jahresdaten2 = jahresdaten2.sort_values(by='Datum', ascending=True) #zur Sicherheit werden die Daten nochmals sortiert
        jahresdaten2.index = jahresdaten2['Datum'] #das Datum wird als Index gesetzt
        y = jahresdaten2['adjustierung_y']
        y = sm.add_constant(y)
        model = RollingOLS(jahresdaten2['adjustierung_x'], y, window=252).fit() #Rollierende Regression über die letzten 252-Handelstage
        jahresdaten2['OLS'] = model.params['adjustierung_y'] #Der Regressions-Parameter wird als eigene Spalte eingesetzt
        jahresdaten2['Erw'] = jahresdaten2['adjustierung_y'] * jahresdaten2['OLS'] #der Erwartungswert für den Close-Price von x wird berechnet
        jahresdaten2['Spread'] = jahresdaten2['adjustierung_x'] - jahresdaten2['Erw'] #die Differenz zwischen tatsächlichem Close und erwartetem Close wird berechnet
        jahresdaten2['Std'] = jahresdaten2['Spread'].rolling(252).std() #vom Spread wird die Standardabweichung rollierend über die letzten 252 Handelstage berechnet
        jahresdaten2['Mean'] = jahresdaten2['Spread'].rolling(252).mean() #und der Durchschnitt des Spreads über die letzten 252 Handelstag
        jahresdaten2['Zscore'] = (jahresdaten2['Spread']-jahresdaten2['Mean'])/jahresdaten2['Std'] #aus diesen Daten wird der Z-Score für jeden Tag berechnet
        jahresdaten2['Zscore_shift'] = jahresdaten2['Zscore'].shift(1) #verschiebt den Z-Score in einer neuen Spalte um einen Tag nach unten        
        jahresdaten2 = jahresdaten2[jahresdaten2['Datum'] > pd.Timestamp(datum3)] #die Daten vor dem Datum der Qualifikation werden gelöscht
        jahresdaten2 = jahresdaten2.copy() #copy-Befehl ist hier notwendig, damit an dieser Kopie anschließend Veränderungen vorgenommen werden können
        jahresdaten2['long'] = np.where((jahresdaten2['Zscore_shift'] < -entry) & (jahresdaten2['Zscore'] >= -entry),1,0) #da wo der Z-Score die negative Eintrittsschwell von unten durchbricht, wird ein long-Signal generiert
        jahresdaten2['short'] = np.where((jahresdaten2['Zscore_shift'] > entry) & (jahresdaten2['Zscore'] <= entry),1,0) #da wo der Z-Score die  Eintrittsschwell von oben durchbricht, wird ein long-Signal generiert
        jahresdaten2['exit'] = (((jahresdaten2['Zscore']<=exit_schwelle) & (jahresdaten2['Zscore_shift']>exit_schwelle)) | ((jahresdaten2['Zscore']>=exit_schwelle) & (jahresdaten2['Zscore_shift']<exit_schwelle)) )*1.0 #wenn der Z-Score die 0 von irgendeiner Seite durchbricht, wird ein Ausstiegssignal generiert
        jahresdaten2.iloc[-1:,[12]] = 1.0 #Exit-Signal am letzten Handelstag
        #Erstellen der Spalten, in welche anschließend unsere aktuelle Positionierung geschrieben wird
        jahresdaten2['long_market'] = 0.0
        jahresdaten2['short_market'] = 0.0
        #diese Variablen tracken die Marktposition während der Iteration
        long_market = 0
        short_market = 0
        #hier wird die Position an jedem einzelnen Tag geprüft
        for i, b in enumerate(jahresdaten2.iterrows()):
            if b[1]['long'] == 1.0:
                long_market = 1
            if b[1]['short'] == 1.0:
                short_market = 1
            if b[1]['exit'] == 1.0:
                long_market = 0
                short_market = 0  
            jahresdaten2.iloc[i, 13] = long_market
            jahresdaten2.iloc[i, 14] = short_market
        jahresdaten2 = jahresdaten2.reset_index(drop=True) #das Datum wird durch eine normale Index ersetzt
        #jahresdaten2['long_market_shift'] = jahresdaten2['long_market'].shift()
        jahresdaten2 = pd.merge(jahresdaten2, dollar[{'Datum', 'Schlusskurs'}], on='Datum') #der Dollarkurs wird an die Daten angeheftet
        jahresdaten3 = jahresdaten2.copy() #der Datensatz wird kopiert
        sym1 = qualpaare.iloc[i,0] #die Variable sym1 wird als Ticker der ersten Aktie des Paars definiert
        sym2 = qualpaare.iloc[i,1] #die Variable sym2 wird als Ticker der zweiten Aktie des Paars definiert
        jahresdaten3.set_index('Datum', drop=True, inplace=True) #im DataFrame jahresdaten3 brauchen wir nun wieder das Datum als Index
        global portfolio #muss gesetzt werden, damit anschließend auch außerhalb der Funktion auf die Variable zugegriffen werden kann
        portfolio = pd.DataFrame(index=jahresdaten3.index) #eine neuer DataFrame mit dem Index von jahresdaten3 wird erstellt
        portfolio['positions'] = jahresdaten3['long_market'] - jahresdaten3['short_market'] #gibt die Position an (-1 für Short, 0 neutral und  für long)
        portfolio[sym1] = 1.0 * jahresdaten3['adjustierung_x']  #gibt die Kapitalentwicklung der ersten Position an
        portfolio[sym2] =  jahresdaten3['adjustierung_y']/ jahresdaten3['Schlusskurs'] #gibt die Kapitalentwicklung der zweiten Position in Euro an
        portfolio['sym1returns'] = portfolio[sym1].pct_change()* portfolio['positions'].shift() #Performance der ersten Position
        portfolio['sym2returns'] = portfolio[sym2].pct_change()* -portfolio['positions'].shift() #Performance der zweiten Position
# Mit der Berechnungsmethode der Performance gibt es noch ein Problem:
# wenn man eine Position zu X Einheiten leer verkauft und diese Position dann schwankt, ergibt ein Verkauf zu Einstandspreis bei
# bei Berechnung mit den negativen täglichen Renditen keine Rendite von null, sondern ein leichter Verlust. 
        portfolio['posisions_value_x'] = (portfolio['sym1returns']+1).cumprod() #kummulierte Performance der ersten Aktie
        portfolio['posisions_value_y'] = (portfolio['sym2returns']+1).cumprod() #kummulierte Performance der ersten Akti
        performance = ((portfolio['posisions_value_x'][-1:]-1)+(portfolio['posisions_value_y'][-1:]-1)).values #die Performance wird aus der letzten Zelle der jeweiligen kummulierten Performance berechnet: Perf_Aktie1 + Perf_Aktie2
        datum = portfolio.index[-1].date() #als Datum für die Kapitalkurve wird der letzte Tag der Handelsphase extrahiert
        ergebnisse1.append(performance[0]) #die Performance wird in die Liste geschrieben
        daten.append(datum) #das Datum wird in die Liste geschrieben
        jahresdaten3['long_market_shift2'] = jahresdaten3['long_market'].shift() #Verschieben der Position um einen Tag nach hinten
        jahresdaten3['short_market_shift2'] = jahresdaten3['short_market'].shift() #Verschieben der Position um einen Tag nach hinten
        jahresdaten3['long_trades_zähler'] = np.where((jahresdaten3['long_market_shift2']==0) & (jahresdaten3['long_market']==1)|(pd.isna(jahresdaten3['long_market_shift2']) & (jahresdaten3['long_market']==1)),1,0) #zählt die tatsächlichen long-Einstiege
        jahresdaten3['short_trades_zähler'] = np.where((jahresdaten3['short_market_shift2']==0) & (jahresdaten3['short_market']==1)|(pd.isna(jahresdaten3['short_market_shift2']) & (jahresdaten3['short_market']==1)),1,0) #zählt die tatsächlichen short-Einstiege
        short_trades_liste.append(jahresdaten3['short_trades_zähler'].sum()) #schreibt die Anzahl der short-Trades in die Liste
        long_trades_liste.append(jahresdaten3['long_trades_zähler'].sum()) #schreibt die Anzahl der long-Trades in die Liste
        einstiegstag_liste.append((np.where((jahresdaten3['short_trades_zähler']==1) | (jahresdaten3['long_trades_zähler']==1)))[0]) #schreibt die Tage, an denen eine Position eingegangen wurde in die Liste
        print(q)
    except:
        print(q,  "Fehler")


#Die einzelnen Listen werden zur Tabelle Auswertung zusammengesetzt:
Auswertung = pd.DataFrame({'Performance': ergebnisse1, 
                           'Datum': daten,
                           'Anzahl Short Trades': short_trades_liste,
                           'Anzahl Long Trades': long_trades_liste,
                           'Einstiegstage': einstiegstag_liste,
                           })     
            


jahresdaten['Log-price St. Jude Medical'] = jahresdaten['CLOSE_y']
jahresdaten['Log-price Fuchs Petrolub in USD'] = jahresdaten['CLOSE_USD_x']

jahresdaten['Log-price St. Jude Medical'].plot()
jahresdaten['Log-price Fuchs Petrolub in USD'].plot()
plt.xlabel('Date')
plt.ylabel('Log-price')
plt.legend()
plt.savefig(r"C:\Users\janhe\Documents\Masterarbeit\Masterarbeit\books_read.png")

ols_result1.resid.plot()
for k in range(43,len(datum)): #len(datum) erster Loop über die Zeitpunkte, startet erst bei 43, da für deutsche Aktien zuvor keine Daten verfügbar sind
    date = datetime.date.fromtimestamp(datum[k]/1000000000) #wandlet den k-ten Eintrag der Liste Datum in ein passendes Datumsformat um. Die 1000... ist nötig, da die Zeit bis auf Millisekunden genau in datum gespeichert. Durch die Division bekommt man nur das Datum, der Rest fällt einfach Weg. Das ganze ist nötig, damit wir später mit der Variablen date die passenden Kursdaten extrahieren können
    a = hdax_index.at[date, 'Index Constituents'] #liest die Mitglieder zum Zeitpunkt date in die Variable, at = Befehl zur Lokalisierung
    a = a.replace("'","") #löscht unnötige Sachen
    a = a.replace("[","")
    a = a.replace("]","")
    a = a.replace(" ","")
    members_hdax = a.split(",") #trennt die Variabel a (in diesem Fall die Mitglieder) bei jedem Komma in ein Listenelement
    
    b = sp500_index.at[date, 'Index Constituents'] #liest die Mitglieder zum Zeitpunkt date in die Variable
    b = b.replace("'","") #löscht unnötige Sachen
    b = b.replace("[","")
    b = b.replace("]","")
    b = b.replace(" ","")
    members_sp500 = b.split(",") #trennt die Variabel b (in diesem Fall die Mitglieder) bei jedem Komma in ein Listenelement

    for i in range(len(members_hdax)): #loop über die Aktien im HDAX
        try: # starte Schleife: wenns eine Fehlermeldung gibt, macht er except --> Achtung: Kann man nicht über Stoptaste stoppen, nur über Kernel-Neustart
            daten_3 = alle_daten_adj[members_hdax[i]] #zieht die Kurse der jeweiligen Aktie aus dem Dictionary
            for j in range(len(members_sp500)): #loop über Aktien des SP500
                    try:
                        daten_4 = alle_daten_adj[members_sp500[j]] #zieht die Aktie des SP500
                        daten_4['Datum'] = pd.to_datetime(daten_4['Datum']) #wandelt die Spalte 'Datum' ins Datumsformat um
                        data = pd.merge(daten_3, daten_4, on='Datum') #die zwei Zeitreihen werden in einem Dataframe vereint
                        data = data.sort_values(by='Datum', ascending=True) #sortiert die Liste nach Datum absteigend
                        data['Index'] = data.index #der Index wird als eigene Spalte in den Dataframe eingefügt
                        monatsanfang = data.groupby(pd.DatetimeIndex(data.Datum).to_period('M')).nth(0) #sucht jeweils den ersten Handelstag im Monat
                        start = monatsanfang.loc[monatsanfang['Datum'] > pd.Timestamp(date)][:1]['Index'] #Wenn z.B. heute der 15.7. ist, mache ich den Kointegrationstest bis zum 30.6. --> Der Befehl führt mich jetzt zum 30.6.
                        data = data.sort_values(by='Datum', ascending=False) #sortiert die Liste nach Datum absteigend
                        jahresdaten = data[start[0]+1:start[0]+253] #extrahiert die Kursdaten der letzten 252 Handelstage ausgehend von start
                        if adfuller(jahresdaten['Log-Preis adjustiert_y'])[1]>0.05 and adfuller(jahresdaten['Log-Preis adjustiert_x'])[1]>0.05: # Test auf Nichtstationarität (p>0.05 = nicht stationär) --> beide Zeitreihen sollten nichtstationär sein, weil
                             erg = coint(jahresdaten['Log-Preis adjustiert_x'], jahresdaten['Log-Preis adjustiert_y']) #berechnet den Kointegrationswert nach Engle-Granger
                             erg_gedreht = coint(jahresdaten['Log-Preis adjustiert_y'], jahresdaten['Log-Preis adjustiert_x']) #da der Test nach Engle-Granger Reihenfolge-Sensitiv ist, wird der Cointegrationswert nochmal für die andere Reihenfolge berechnet
                             list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':erg[1], 'Wert_gedreht':erg_gedreht[1]}) #die Ergebnisse werden in die Liste eingehängt
                             print(k, i, j)
                        elif adfuller(jahresdaten['Log-Preis adjustiert_y'])[1]<0.05 and adfuller(jahresdaten['Log-Preis adjustiert_x'])[1]<0.05: #wenn beide Zeitreihen stationär sind, werden sie mit einem sehr niedrigen p-Wert in die Liste geschrieben, da das die besten Kanditaten sind
                            list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':-1, 'Wert_gedreht':-1})
                            print(k, i,j, -1)
                        else:
                            list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':999999, 'Wert_gedreht':999999}) #wenn eine Zeitreihe nicht-Stationär ist und die andere schon, wird das Paar mit 999999 in die Liste aufgenommen 
                            print(k, i,j, 999999)
                    except:
                        list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':111111, 'Wert_gedreht':111111}) #wenn es Probleme mit den Daten bei der SP500-Aktie gibt (z.B. fehlende Kurse) wird das Paar mit einem hohen Wert in die Liste aufgenommen, sodass es später aussortiert wird
                        print(k, i,j, 111111)
        except:
            list.append({'Aktie1':members_hdax[i], 'Aktie2':members_sp500[j], 'Datum':date, 'Wert':222222, 'Wert_gedreht':222222}) #wenn es Probleme mit den Daten bei der HDAX-Aktie gibt (z.B. fehlende Kurse) wird das Paar mit einem hohen Wert in die Liste aufgenommen, sodass es später aussortiert wird
            print(k, i,j, 222222)

zwspeicher.append(list) #überführe Liste in Dataframe

zwspeicher = pd.read_csv(r"C:\Users\janhe\Documents\Masterarbeit\Masterarbeit\Ausgabe_komplett_final.csv)
alle_qualpaare = zwspeicher[(zwspeicher['Wert'] < 0.05) & (zwspeicher['Wert_gedreht'] < 0.05)] #es werden nur Paare mit einem p-Wert kleiner 0.05 in beiden Richtungen beachtet
#zurechtstutzen der qualifizierten Paare (nur bis Mitte 2019, damit noch ein Jahr getradet werden kann und nur ab September 2013, da vorher Daten für deutsche Werte fehlen)
qualpaare = alle_qualpaare[alle_qualpaare['Datum'] >= str(datetime.date(2012,8,30)) ] #datetime.date(2018,12,31)
qualpaare = qualpaare[qualpaare['Datum'] < str(datetime.date(2019,11,30))] #datetime.date(2018,12,31)



#Backtest: https://www.quantstart.com/articles/Backtesting-An-Intraday-Mean-Reversion-Pairs-Strategy-Between-SPY-And-IWM/

ergebnisse1 = [] #eine leere Liste für die Performance jedes einzelnen Trades wird erstellt
daten = [] #eine leere Liste für das Enddatum jedes einzelnen Trades wird erstellt
short_trades_liste = [] #eine leere Liste für den Trade-Zähler wird erstellt
long_trades_liste = [] #eine leere Liste für den Trade-Zähler wird erstellt
einstiegstag_liste = [] #eine leere Liste für das Einstiegsdatum jedes einzelnen Trades wird erstellt

entry=2 #definiert die Einsteigsschwelle des Z-Scores
exit_schwelle=0  #definiert die Ausstiegsschwelle des Z-Scores 
anzahl_nicht_geöffnet=0 #setzt die Startwerte für die Zähler nicht-geöffneter Paare auf 0
anzahl_geöffnet=0 #setzt die Startwerte für die Zähler geöffneter Paare auf 0

q=1500

#Loop über die qualifizierten Paare und Berechnung der Performance
for q in range(len(qualpaare)):   
    try:
        stock1 = alle_daten_adj[qualpaare.iloc[q,0]] #zieht sich zuerst die kurs-Daten des Paares
        stock2 = alle_daten_adj[qualpaare.iloc[q,1]] #zieht sich zuerst die kurs-Daten des Paares
        stock1 = stock1.copy() #copy-Befehl ist hier notwendig, damit an dieser Kopie anschließend Veränderungen vorgenommen werden können
        stock1['Datum'] = pd.to_datetime(stock1.Datum) #formatiert die Spalte Datum ins Datumsformat
        stock2 = stock2.copy() #copy-Befehl ist hier notwendig, damit an dieser Kopie anschließend Veränderungen vorgenommen werden können
        stock2['Datum'] = pd.to_datetime(stock2.Datum) #formatiert die Spalte Datum ins Datumsformat
        datum3 = qualpaare.iloc[q,2] #zieht das Datum, zu der das Paar qualifiziert wurde
        data2 = pd.merge(stock1, stock2, on='Datum') #die zwei Zeitreihen werden in einem Dataframe vereint
        data2['Index'] = data2.index #der Index wird als eigene Spalte in den Dataframe eingefügt
        data2 = data2.sort_values(by='Datum', ascending=True) #sortiert die Liste nach Datum
        monatsanfang = data2.groupby(pd.DatetimeIndex(data2.Datum).to_period('M')).nth(0) #sucht jeweils den ersten Handelstag im Monat
        start = monatsanfang.loc[monatsanfang['Datum'] > pd.Timestamp(datum3)].head(1)['Index'] #gibt den nächsten ersten Handelstag ausgehend von date aus
        global jahresdaten2 #muss gesetzt werden, damit anschließend auch außerhalb der Funktion auf die Variable zugegriffen werden kann
        data2 = data2.sort_values(by='Datum', ascending=False) #erneutes sortieren der Liste
        jahresdaten2 = data2[start[0]-126:start[0]+505][['Datum', 'adjustierung_x', 'adjustierung_y']] #zieht sich die Daten der letzten zwei Jahre und der folgenden sechs Monate
        jahresdaten2 = jahresdaten2.sort_values(by='Datum', ascending=True) #zur Sicherheit werden die Daten nochmals sortiert
        jahresdaten2.index = jahresdaten2['Datum'] #das Datum wird als Index gesetzt
        model = RollingOLS(jahresdaten2['adjustierung_x'],jahresdaten2['adjustierung_y'], window=252).fit() #Rollierende Regression über die letzten 252-Handelstage
        jahresdaten2['OLS'] = model.params #Der Regressions-Parameter wird als eigene Spalte eingesetzt
        jahresdaten2['Erw'] = jahresdaten2['adjustierung_y'] * jahresdaten2['OLS'] #der Erwartungswert für den Close-Price von x wird berechnet
        jahresdaten2['Spread'] = jahresdaten2['adjustierung_y'] - jahresdaten2['Erw'] #die Differenz zwischen tatsächlichem Close und erwartetem Close wird berechnet
        jahresdaten2['Std'] = jahresdaten2['Spread'].rolling(252).std() #vom Spread wird die Standardabweichung rollierend über die letzten 252 Handelstage berechnet
        jahresdaten2['Mean'] = jahresdaten2['Spread'].rolling(252).mean() #und der Durchschnitt des Spreads über die letzten 252 Handelstag
        jahresdaten2['Zscore'] = (jahresdaten2['Spread']-jahresdaten2['Mean'])/jahresdaten2['Std'] #aus diesen Daten wird der Z-Score für jeden Tag berechnet
        jahresdaten2['Zscore_shift'] = jahresdaten2['Zscore'].shift(1) #verschiebt den Z-Score in einer neuen Spalte um einen Tag nach unten        
        jahresdaten2 = jahresdaten2[jahresdaten2['Datum'] > pd.Timestamp(datum3)] #die Daten vor dem Datum der Qualifikation werden gelöscht
        jahresdaten2 = jahresdaten2.copy() #copy-Befehl ist hier notwendig, damit an dieser Kopie anschließend Veränderungen vorgenommen werden können
        jahresdaten2['long'] = np.where((jahresdaten2['Zscore_shift'] < -entry) & (jahresdaten2['Zscore'] >= -entry),1,0) #da wo der Z-Score die negative Eintrittsschwell von unten durchbricht, wird ein long-Signal generiert
        jahresdaten2['short'] = np.where((jahresdaten2['Zscore_shift'] > entry) & (jahresdaten2['Zscore'] <= entry),1,0) #da wo der Z-Score die  Eintrittsschwell von oben durchbricht, wird ein long-Signal generiert
        jahresdaten2['exit'] = (((jahresdaten2['Zscore']<=exit_schwelle) & (jahresdaten2['Zscore_shift']>exit_schwelle)) | ((jahresdaten2['Zscore']>=exit_schwelle) & (jahresdaten2['Zscore_shift']<exit_schwelle)) )*1.0 #wenn der Z-Score die 0 von irgendeiner Seite durchbricht, wird ein Ausstiegssignal generiert
        jahresdaten2.iloc[-1:,[12]] = 1.0 #Exit-Signal am letzten Handelstag
        #Erstellen der Spalten, in welche anschließend unsere aktuelle Positionierung geschrieben wird
        jahresdaten2['long_market'] = 0.0
        jahresdaten2['short_market'] = 0.0
        #diese Variablen tracken die Marktposition während der Iteration
        long_market = 0
        short_market = 0
        #hier wird die Position an jedem einzelnen Tag geprüft
        for i, b in enumerate(jahresdaten2.iterrows()):
            if b[1]['long'] == 1.0:
                long_market = 1
            if b[1]['short'] == 1.0:
                short_market = 1
            if b[1]['exit'] == 1.0:
                long_market = 0
                short_market = 0  
            jahresdaten2.iloc[i, 13] = long_market
            jahresdaten2.iloc[i, 14] = short_market
        jahresdaten2 = jahresdaten2.reset_index(drop=True) #das Datum wird durch eine normale Index ersetzt
        #jahresdaten2['long_market_shift'] = jahresdaten2['long_market'].shift()
        jahresdaten2 = pd.merge(jahresdaten2, dollar[{'Datum', 'Schlusskurs'}], on='Datum') #der Dollarkurs wird an die Daten angeheftet
        jahresdaten3 = jahresdaten2.copy() #der Datensatz wird kopiert
        sym1 = qualpaare.iloc[i,0] #die Variable sym1 wird als Ticker der ersten Aktie des Paars definiert
        sym2 = qualpaare.iloc[i,1] #die Variable sym2 wird als Ticker der zweiten Aktie des Paars definiert
        jahresdaten3.set_index('Datum', drop=True, inplace=True) #im DataFrame jahresdaten3 brauchen wir nun wieder das Datum als Index
        global portfolio #muss gesetzt werden, damit anschließend auch außerhalb der Funktion auf die Variable zugegriffen werden kann
        portfolio = pd.DataFrame(index=jahresdaten3.index) #eine neuer DataFrame mit dem Index von jahresdaten3 wird erstellt
        portfolio['positions'] = jahresdaten3['long_market'] - jahresdaten3['short_market'] #gibt die Position an (-1 für Short, 0 neutral und  für long)
        portfolio[sym1] = 1.0 * jahresdaten3['adjustierung_x']  #gibt die Kapitalentwicklung der ersten Position an
        portfolio[sym2] =  jahresdaten3['adjustierung_y']/ jahresdaten3['Schlusskurs'] #gibt die Kapitalentwicklung der zweiten Position in Euro an
        portfolio['sym1returns'] = portfolio[sym1].pct_change()* portfolio['positions'].shift() #Performance der ersten Position
        portfolio['sym2returns'] = portfolio[sym2].pct_change()* -portfolio['positions'].shift() #Performance der zweiten Position
# Mit der Berechnungsmethode der Performance gibt es noch ein Problem:
# wenn man eine Position zu X Einheiten leer verkauft und diese Position dann schwankt, ergibt ein Verkauf zu Einstandspreis bei
# bei Berechnung mit den negativen täglichen Renditen keine Rendite von null, sondern ein leichter Verlust. 
        portfolio['posisions_value_x'] = (portfolio['sym1returns']+1).cumprod() #kummulierte Performance der ersten Aktie
        portfolio['posisions_value_y'] = (portfolio['sym2returns']+1).cumprod() #kummulierte Performance der ersten Akti
        performance = ((portfolio['posisions_value_x'][-1:]-1)+(portfolio['posisions_value_y'][-1:]-1)).values #die Performance wird aus der letzten Zelle der jeweiligen kummulierten Performance berechnet: Perf_Aktie1 + Perf_Aktie2
        datum = portfolio.index[-1].date() #als Datum für die Kapitalkurve wird der letzte Tag der Handelsphase extrahiert
        ergebnisse1.append(performance[0]) #die Performance wird in die Liste geschrieben
        daten.append(datum) #das Datum wird in die Liste geschrieben
        jahresdaten3['long_market_shift2'] = jahresdaten3['long_market'].shift() #Verschieben der Position um einen Tag nach hinten
        jahresdaten3['short_market_shift2'] = jahresdaten3['short_market'].shift() #Verschieben der Position um einen Tag nach hinten
        jahresdaten3['long_trades_zähler'] = np.where((jahresdaten3['long_market_shift2']==0) & (jahresdaten3['long_market']==1)|(pd.isna(jahresdaten3['long_market_shift2']) & (jahresdaten3['long_market']==1)),1,0) #zählt die tatsächlichen long-Einstiege
        jahresdaten3['short_trades_zähler'] = np.where((jahresdaten3['short_market_shift2']==0) & (jahresdaten3['short_market']==1)|(pd.isna(jahresdaten3['short_market_shift2']) & (jahresdaten3['short_market']==1)),1,0) #zählt die tatsächlichen short-Einstiege
        short_trades_liste.append(jahresdaten3['short_trades_zähler'].sum()) #schreibt die Anzahl der short-Trades in die Liste
        long_trades_liste.append(jahresdaten3['long_trades_zähler'].sum()) #schreibt die Anzahl der long-Trades in die Liste
        einstiegstag_liste.append((np.where((jahresdaten3['short_trades_zähler']==1) | (jahresdaten3['long_trades_zähler']==1)))[0]) #schreibt die Tage, an denen eine Position eingegangen wurde in die Liste
        print(q)
    except:
        print(q,  "Fehler")


#Die einzelnen Listen werden zur Tabelle Auswertung zusammengesetzt:
Auswertung = pd.DataFrame({'Performance': ergebnisse1, 
                           'Datum': daten,
                           'Anzahl Short Trades': short_trades_liste,
                           'Anzahl Long Trades': long_trades_liste,
                           'Einstiegstage': einstiegstag_liste,
                           })

Auswertung3 = Auswertung.fillna(0) #Auswertung wird in Auswertung3 überführt, um die usrpüngliche Tabelle nicht zu überschreiben
#Die Trades ohne Performance werden aussortiert, um den Mean für jeden Monat zu bekommen:
idx = Auswertung3.index[Auswertung3['Performance'] == 0].tolist() 
Auswertung3 = Auswertung3.loc[Auswertung3.index[Auswertung3['Performance'] != 0].tolist()]
Auswertung3 = Auswertung3.set_index('Datum')
Auswertung3.index = pd.to_datetime(Auswertung3.index)
#Die Trades werden nach Monat gruppiert, um eine durchschnittliche Performance für jeden tatsächlichen Trade zu bekommen. 
Auswertung3 = Auswertung3.groupby(pd.Grouper(freq="M")).agg({'Performance':'mean', 'Anzahl Short Trades': 'count'}).rename(columns={'Anzahl Short Trades':'Anzahl'})
#Aus diesem Monatsmittelwert kann dann die Kapitalkurve gezeichnet werden
(Auswertung3['Performance']+1).cumprod().plot()