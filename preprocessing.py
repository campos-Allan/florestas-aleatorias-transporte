"""formatting dataframes
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np


def passenger_formatting(df_trips: pd.DataFrame) -> pd.DataFrame:
    """formatting dataframe with trips, generating new atributes

    Args:
        df_trips (pd.DataFrame): trips dataframe

    Returns:
        pd.DataFrame: formatted dataframe
    """

    # new data attributes
    df_trips['Year'] = df_trips['trip'].map(lambda x: x.year)
    df_trips['Month'] = df_trips['trip'].map(lambda x: x.month)
    df_trips['Day'] = df_trips['trip'].map(lambda x: x.day)
    df_trips['Day of the Week'] = df_trips['trip'].map(
        lambda x: x.weekday())
    df_trips['Week of the Year'] = df_trips['trip'].apply(
        lambda x: x.weekofyear)

    # new holiday attributes
    df_trips['Pre Holiday'] = 0
    df_trips['Holiday'] = 0
    df_trips['After Holiday'] = 0
    holidays = [date(2016, 1, 1), date(2016, 2, 8), date(2016, 2, 9),
                date(2016, 2, 10), date(2016, 3, 25), date(2016, 4, 21),
                date(2016, 5, 1), date(2016, 5, 26), date(2016, 8, 15),
                date(2016, 9, 7), date(2016, 10, 12), date(2016, 11, 2),
                date(2016, 11, 15), date(2016, 12, 8), date(2016, 12, 25),
                date(2017, 1, 1), date(2017, 2, 27), date(2017, 2, 28),
                date(2017, 3, 1), date(2017, 4, 14), date(2017, 4, 21),
                date(2017, 5, 1), date(2017, 6, 15), date(2017, 8, 15),
                date(2017, 9, 7), date(2017, 10, 12), date(2017, 11, 2),
                date(2017, 11, 15), date(2017, 12, 8), date(2017, 12, 25),
                date(2018, 1, 1), date(2018, 2, 12), date(2018, 2, 13),
                date(2018, 2, 14), date(2018, 3, 30), date(2018, 4, 21),
                date(2018, 5, 1), date(2018, 5, 31), date(2018, 8, 15),
                date(2018, 9, 7), date(2018, 10, 12), date(2018, 11, 2),
                date(2018, 11, 15), date(2018, 12, 8), date(2018, 12, 25),
                date(2019, 1, 1), date(2019, 3, 4), date(2019, 3, 5),
                date(2019, 3, 6), date(2019, 4, 19), date(2019, 4, 21),
                date(2019, 5, 1), date(2019, 6, 20), date(2019, 8, 15),
                date(2019, 9, 7), date(2019, 10, 12), date(2019, 11, 2),
                date(2019, 11, 15), date(2019, 12, 8), date(2019, 12, 25),
                date(2020, 1, 1), date(2020, 2, 24), date(2020, 2, 25),
                date(2020, 2, 26), date(2020, 4, 10), date(2019, 4, 21),
                date(2020, 5, 1), date(2020, 6, 11), date(2020, 8, 15),
                date(2020, 9, 7), date(2020, 10, 12), date(2020, 11, 2),
                date(2020, 11, 15), date(2020, 12, 8), date(2020, 12, 25),
                date(2021, 1, 1), date(2021, 2, 15), date(2021, 2, 16),
                date(2021, 2, 17), date(2021, 4, 2), date(2021, 4, 21),
                date(2021, 5, 1), date(2021, 6, 3), date(2021, 8, 15),
                date(2021, 9, 7), date(2021, 10, 12), date(2021, 11, 2),
                date(2021, 11, 15), date(2021, 12, 8), date(2021, 12, 25)]
    df_trips['Holiday'] = df_trips['trip'].apply(
        lambda x: 1 if x in holidays else 0)
    day = timedelta(1)
    pre_holiday = []
    after_holiday = []
    for i in holidays:
        pre_holiday.append(i-day)
        after_holiday.append(i+day)
    df_trips['Pre Holiday'] = df_trips['trip'].apply(
        lambda x: 1 if x in pre_holiday else 0)
    df_trips['After Holiday'] = df_trips['trip'].apply(
        lambda x: 1 if x in after_holiday else 0)

    # new pandemic attributes
    df_trips['Pandemic'] = df_trips["trip"].map(
        lambda x: 1 if x > date(2020, 3, 15) else 0)

    # replacing bus line names for integers from 1 to 319 (number of bus lines)
    df_trips['bus_line'] = df_trips['bus_line'].astype(str)
    df_trips['bus_line'] = df_trips['bus_line'].replace(
        list(df_trips['bus_line'].unique()), list(range(1, 320)))
    return df_trips


def weather_formatting(df_rain_temperature: pd.DataFrame) -> pd.DataFrame:
    """formatting weather dataframe

    Args:
        df_clima (pd.DataFrame): dataframe to be formatted

    Returns:
        pd.DataFrame: final dataframe
    """
    df_rain_temperature['Rain'] = df_rain_temperature['Rain'].str.replace(
        ",", ".")
    df_rain_temperature['Temp'] = df_rain_temperature['Temp'].str.replace(
        ",", ".")

    df_rain_temperature['Rain'] = df_rain_temperature['Rain'].replace(
        "-9999", np.nan)
    df_rain_temperature['Temp'] = df_rain_temperature['Temp'].replace(
        "-9999", np.nan)

    df_rain_temperature['Rain'] = df_rain_temperature['Rain'].astype(float)
    df_rain_temperature['Temp'] = df_rain_temperature['Temp'].astype(float)

    df_rain_temperature['Rain'] = df_rain_temperature['Rain'].fillna(0)
    df_rain_temperature['Temp'] = df_rain_temperature['Temp'].fillna(
        df_rain_temperature['Temp'].mean())

    # groupby hourly measures by day, and making new attributes, such as
    # total rain in day; mean, max and min temperature in day
    df_rain_temperature = df_rain_temperature.groupby('Data', axis=0, as_index=False).agg(
        {'Rain': 'sum', 'Temp': ['mean', 'max', 'min']})
    df_rain_temperature.columns = df_rain_temperature.columns.droplevel(0)

    # making compatibility possible between weather dataframe and trip dataframe
    # by using similar sorted data column named trip
    df_rain_temperature[''] = pd.to_datetime(
        df_rain_temperature[''], dayfirst=True)
    df_rain_temperature = df_rain_temperature.sort_values(by='')
    df_rain_temperature = df_rain_temperature.rename(columns={'': 'trip'})

    return df_rain_temperature
