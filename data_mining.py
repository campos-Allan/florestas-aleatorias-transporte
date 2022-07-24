"""data compiler, read main.py disclaimer to know why
passenger_data_compiler call is in comments
"""
import os
import pandas as pd


def passenger_data_compiler(
        df_trips: pd.DataFrame) -> pd.DataFrame:
    """compiling spreadsheets into a dataframe
    Args:
        df_trips (pd.DataFrame): all the trip data
    Returns:
        pd.DataFrame: dataframe with all spreadsheet data
    """
    year = 16
    month = 1
    file_location = os.getcwd().replace('\\', '/') + \
        '/passenger_data'
    while year < 22:
        if month < 10:
            spreadsheet_path = file_location + \
                '/mco-0'+str(month)+'-20'+str(year)+".csv"
        else:
            spreadsheet_path = file_location + \
                '/mco-'+str(month)+'-20'+str(year)+".csv"

        df_spreadsheet = pd.DataFrame(pd.read_csv(spreadsheet_path, encoding='utf-8', sep=';',
                                                  usecols=[' VIAGEM', ' LINHA',
                                                           ' CATRACA SAIDA', ' CATRACA CHEGADA']))
        df_spreadsheet = df_spreadsheet.rename(
            {' VIAGEM': 'trip', ' LINHA': 'bus_line',
             ' CATRACA SAIDA': 'gate_departure', ' CATRACA CHEGADA': 'gate_arrival'}, axis=1)

        # subtracting values from ticket gate at the arrival minus its value from departure
        # to get a passenger number in the individual trip
        # sometimes value from ticket gate is smaller in arrival comparing with departure
        # this happens because the gate can only register up to 5 digits, so when it reaches
        # '99999', the gate resets, thats why theres condition in lambda
        df_spreadsheet['passenger'] = df_spreadsheet.apply(lambda x: (
            100000-x[2]+x[3]) if (((x[3]-x[2]) < 0) & ((x[2]/100) >= 998)) else (x[3]-x[2]), axis=1)
        df_spreadsheet = df_spreadsheet.drop(
            ['gate_departure', 'gate_arrival'], axis=1)
        df_trips = pd.concat(
            [df_spreadsheet, df_trips], ignore_index=True)

        print(str(month)+'-20'+str(year)+' OK')
        month += 1
        if month > 12:
            year += 1
            month = 1
    return df_trips


def weather_data_compiler(
        df_rain_temperature: pd.DataFrame) -> pd.DataFrame:
    """compile weather spreasheets into a dataframe
    spreadsheets from INMET https://bdmep.inmet.gov.br/
    Args:
        df_rain_temperature (pd.DataFrame): data goes here
    Returns:
        pd.DataFrame: final dataframe with all weather data
    """

    # spreadsheets from INMET https://bdmep.inmet.gov.br/
    year = 16
    file_directory = os.getcwd().replace('\\', '/')+'/weather_data'
    while year < 22:
        path = file_directory+'/mg'+str(year)+'.csv'
        df_spreadsheet = pd.DataFrame(pd.read_csv(
            path, encoding='utf-8', sep=';'))
        df_rain_temperature = pd.concat(
            [df_spreadsheet, df_rain_temperature], ignore_index=True)
        year += 1
    return df_rain_temperature
