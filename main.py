""" undergraduate thesis project - predicting demand with random forest
in a real public transportation database

-> run the code through here, for more info README.md
"""
import pandas as pd
import graphics
import random_forests
import preprocessing
import data_mining

df_passenger = pd.DataFrame()

# compiling passenger data -> already on final_passenger_data.csv, read below

'''
DISCLAIMER: the city that was used on this project
took out passenger data from its website, so those files are not available
anymore for web scraping, about 5GB of spreadsheets,
going every month from 2016 to 2021 with every bus trip, i got the files
on my HD but they are too big, so the compiled result from this part of the code
is the final_passenger_data.csv spreadsheet, after groupby date and bus line

df_passenger = data_mining.passenger_data_compiler(df_passenger)
'''
df_passenger = pd.DataFrame(pd.read_csv(
    'final_passenger_data.csv', encoding='utf-8', sep=';',
    usecols=['trip', 'bus_line', 'passenger']))

df_passenger['trip'] = pd.to_datetime(
    df_passenger['trip'], dayfirst=True)
df_passenger = df_passenger.sort_values(by='trip')

'''
# generating graphic of all trips, before and after groupby
graphics.passenger_alltrips_graphic(df_passenger)
df_passenger = df_passenger.groupby(
    ['trip', 'bus_line'], axis=0, as_index=False).sum()
'''
# only possible to get the graphic after groupby because of the disclaimer above
graphics.passenger_alltrips_graphic(df_passenger)

# formatting passenger dataframe
df_passenger = preprocessing.passenger_formatting(df_passenger)

# compiling and formatting weather data
df_weather = pd.DataFrame()
df_weather = data_mining.weather_data_compiler()
df_weather = preprocessing.weather_formatting(df_weather)

# merge weather dataframe with trip dataframe
df_final_data = pd.merge(df_passenger, df_weather, on='trip')
df_final_data = df_final_data.rename(columns={
    'sum': 'Rain',
    'mean': 'Mean Temperature',
    'max': 'Max Temperature',
    'min': 'Min Temperature'})
df_final_data.drop("trip", axis=1, inplace=True)

# separating feature variables from independent variable
X = df_final_data.copy()
y = X.pop("passenger")

# first experiment with full dataset
tscv = random_forests.TimeSeriesSplit(n_splits=24)  # +- 3 months
Results_fulldataset = pd.DataFrame()
Results_fulldataset = random_forests.hyperparameters(
    tscv, 'TODO', X, y)


# dividing dataset in pre pandemic (2016-2019) and pandemic (2020-2021)
X_pan = X[400038:]
y_pan = y[400038:]

X_pre = X[:400038]
y_pre = y[:400038]

# second experiment with only pre pandemic data
tscv = random_forests.TimeSeriesSplit(n_splits=16)  # +- 3 months
Results_prepandemic_data = pd.DataFrame()
Results_prepandemic_data = random_forests.hyperparameters(
    tscv, 'PRE', X_pre, y_pre)


# third experiment with only pandemic data
tscv = random_forests.TimeSeriesSplit(n_splits=8)  # +- 3 months
Results_pandemic_data = pd.DataFrame()
Results_pandemic_data = random_forests.hyperparameters(
    tscv, 'PAN', X_pan, y_pan)

# these other experiments were made to see effects of trend,
# seasonality and temporality on this time series

# generating final graphics with performance measurements in each experiment
graphics.model_graphics(Results_fulldataset)
graphics.model_graphics(Results_prepandemic_data)
graphics.model_graphics(Results_pandemic_data)
