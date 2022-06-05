"""machine learning models
"""

from typing import List
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import graphics


def hyperparameters(
        time_split: TimeSeriesSplit,
        instance: str,
        x_features: pd.DataFrame, y_passengers: pd.Series) -> pd.DataFrame:
    """ modelos de floresta aleatória com diferentes hiperparâmetros
    sendo iterados na função anterior

    Args:
        time_split (TimeSeriesSplit): growing-window forward-validation method
        instance (str): todo -> all dataset; pan -> pandemic; pre -> pre pandemic
        x_features (pd.DataFrame): attributes
        y_passengers (pd.Series): independent variable

    Returns:
        pd.DataFrame: performance metrics
    """
    # different configs for node and max depth in random forest hyperparameters
    trees = [2, 10, 20, 50, 100]
    depth = [5, 10, 20, 50, 100]
    results = pd.DataFrame()
    for i in trees:
        for j in depth:
            results[instance+' DATAFRAME, '+'A' +
                    str(i)+' N'+str(j)] = timesplit(time_split, x_features, y_passengers, i, j)
    return results


def timesplit(
        time_split: TimeSeriesSplit,
        x_features: pd.DataFrame, y_passengers: pd.Series,
        n_estimators: int, max_depth: int) -> List[List]:
    """dividing dataset between traning and testing with iterations going
    through data with the method growing-window forward-validation

    Args:
        time_split (TimeSeriesSplit): growing-window forward-validation model
        x_features (pd.DataFrame): attributes
        y_passengers (pd.Series): independent variable
        n_estimators (int): number of trees in random forest
        max_depth (int): maximum depth of trees in random forest

    Returns:
        List[List]: performance metrics
    """
    reg = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    results = []
    for fold, (train_index, test_index) in enumerate(time_split.split(x_features)):
        start = time.time()  # time spent during each iteration

        # train/test index changing according to the iteration
        # of the growing-window with foward validation
        print(f"Fold: {fold}")
        print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
        print("\n")

        # train/test division on x and y variables
        x_train, x_test = x_features.iloc[train_index], x_features.iloc[test_index]
        y_train, y_test = y_passengers.iloc[train_index], y_passengers.iloc[test_index]

        reg.fit(x_train, y_train)
        acc = round(reg.score(x_test, y_test) * 100, 2)  # R²
        y_pred = reg.predict(x_test)
        erro_ab = mean_absolute_error(
            y_test, y_pred)

        end = time.time()

        results.append([acc, erro_ab, train_index[-1], (end-start)])
        print("Random Forest R2:", acc, "%")
        # scatter plot with all the predictions vs real values (called per iteration)
        graphics.realvalue_vs_prediction_graphic(
            y_test, y_pred, x_train, x_features, max_depth, n_estimators)

    return results
