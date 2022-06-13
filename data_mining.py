"""data compiler, read main.py disclaimer to know why
passenger_data_compiler call is in comments
"""
import glob
import os
import pandas as pd
import numpy as np


def passenger_data_compiler(
        df_trips: pd.DataFrame) -> pd.DataFrame:
    """compiling spreadsheets into a dataframe
    Args:
        df_trips (pd.DataFrame): starting dataframe

    Returns:
        pd.DataFrame: dataframe with all spreadsheet data
    """
    file_location = os.getcwd().replace('\\', '/') + \
        '/passenger_data'  # name of folder with spreadsheets
    csv_files = glob.glob(file_location + "/*.csv")
    df_files = (pd.read_csv(file,
                            encoding='utf-8', sep=';',
                            usecols=[' VIAGEM', ' LINHA',
                                     ' CATRACA SAIDA', ' CATRACA CHEGADA'],
                            dtype={' VIAGEM': object, ' LINHA': object,
                                   ' CATRACA SAIDA': np.int64,
                                   ' CATRACA CHEGADA': np.int64}) for file in csv_files)

    df_spreadsheet = pd.concat(df_files, ignore_index=True)
    df_spreadsheet = df_spreadsheet.rename(
        {' VIAGEM': 'trip', ' LINHA': 'bus_line',
         ' CATRACA SAIDA': 'gate_departure', ' CATRACA CHEGADA': 'gate_arrival'}, axis=1)

    df_spreadsheet['passenger'] = df_spreadsheet['gate_arrival'] - \
        df_spreadsheet['gate_departure']
    # taking off anomalous values, like trips with >500 passenger
    # also trips with negative and zero values (likely from typing errors)
    df_spreadsheet = df_spreadsheet[(df_spreadsheet['passenger'] < 500) & (
        df_spreadsheet['passenger'] > 0)]

    df_spreadsheet.drop(
        ['gate_departure', 'gate_arrival'], axis=1, inplace=True)
    df_trips = pd.concat([df_spreadsheet, df_trips], ignore_index=True)
    return df_trips


def weather_data_compiler() -> pd.DataFrame:
    """compile weather spreasheets into a dataframe
    spreadsheets from INMET https://bdmep.inmet.gov.br/

    Returns:
        pd.DataFrame: final dataframe with all weather data
    """

    # name of folder with spreadsheets
    file_location = os.getcwd().replace('\\', '/')+'/weather_data'
    csv_files = glob.glob(file_location + "/*.csv")
    df_files = (pd.read_csv(file,
                            encoding='utf-8', sep=';',
                            dtype={'Data': object, 'Hour': object,
                                   'Rain': object,
                                   'Temp': object}) for file in csv_files)
    df_rain_temperature = pd.concat(df_files, ignore_index=True)

    return df_rain_temperature
