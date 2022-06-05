"""graphic generator
"""

from typing import Any, List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def passenger_alltrips_graphic(data: pd.DataFrame) -> None:
    """scatter plot all trips, before and after groupby per data & bus line

    Args:
        data (pd.DataFrame): dataframe with trips
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)
    fig, axis = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(fig, left=True, bottom=True)
    sns.scatterplot(x="trip", y="passenger",
                    palette="ch:r=-.2,d=.3_r",
                    sizes=(1, 8), linewidth=0,
                    data=data, ax=axis)
    plt.show()


def realvalue_vs_prediction_graphic(
        y_test: Any, y_pred: Any,
        x_train: pd.Series, x_features: pd.DataFrame,
        max_depth: int, n_estimators: int) -> None:
    """scatter plot comparing predictions vs real values
    for each iteration of each dataset experiment
    """
    df_comparison = pd.DataFrame({'Real Value': y_test, 'Prediction': y_pred})
    plt.scatter(y_test, y_pred)

    sns.regplot(x="Real Value", y="Prediction",
                data=df_comparison, color='red', marker="")

    # saving graphics of each experiment
    if len(x_features) > 410000:
        plt.savefig('realvalue_vs_prediction/TODO DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(x_train))+'.png')
    elif len(x_features) < 200000:
        plt.savefig('realvalue_vs_prediction/PAN DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(x_train))+'.png')
    else:
        plt.savefig('realvalue_vs_prediction/PRE DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(x_train))+'.png')
    plt.clf()


def best_model_graphics(results: List[List]) -> None:
    """generating graphics of peformance metrics from
    model with hyperparameters tuned to max


    Args:
        resultados (List[List]): perfomance metrics
    """
    results_r2 = results.applymap(lambda x: x[0])
    results_mae = results.applymap(lambda x: x[1])
    results_time = results.applymap(lambda x: x[3])

    fig, ax1 = plt.subplots()
    fig.suptitle('A100 N100')  # A= Nodes; N= Max Depth
    ax1.plot(results_r2.iloc[:, -1], color="red", marker="o")
    ax1.set_xlabel("Window Iterations", fontsize=14)
    ax1.set_ylabel("R²", color="red", fontsize=14)
    ax2 = ax1.twinx()
    ax2.plot(results_mae.iloc[:, -1], color="blue", marker="o")
    ax2.set_ylabel("MAE (Passengers)",
                   color="blue", fontsize=14)
    plt.tight_layout()
    plt.show()

    # boxtplotting the result from every random forest model with different
    # hyperparameters
    boxplotter(results_r2, 'box_r2')
    boxplotter(results_mae, 'box_mae')
    boxplotter(results_time, 'plt')


def boxplotter(results: Any, graphic_type: str) -> None:
    """boxtplotting the result from every random forest model
    with different hyperparameters, plus elapsed time

    Args:
        resultados (Any): perfomance metrics
        graphic_type (str): type of graphic: random forest boxplot or time plot
    """
    results.columns = ['A2 N5', 'A2 N10', 'A2 N20', 'A2 N50', 'A2 N100',
                       'A10 N5', 'A10 N10', 'A10 N20', 'A10 N50',
                       'A10 N100', 'A20 N5', 'A20 N10', 'A20 N20',
                       'A20 N50', 'A20 N100', 'A50 N5', 'A50 N10',
                       'A50 N20', 'A50 N50', 'A50 N100', 'A100 N5',
                       'A100 N10', 'A100 N20', 'A100 N50', 'A100 N100']
    sns.set_style("whitegrid")
    if graphic_type == 'plt':
        # elapsed time graphic in each random forest model
        results = results.sum()
        results.plot(fontsize=14)
        plt.xlabel('Random Forest Models')
        plt.ylabel('Seconds')
        plt.tight_layout()
        plt.show()
        plt.clf()
    else:
        # R² and MAE boxplot in each random forest model
        plt.boxplot(results)
        plt.xticks(ticks=range(len(results.columns)),
                   labels=results.columns, rotation=45)
        plt.xlabel('Random Forest Models')
        if graphic_type == 'box_r2':
            plt.ylabel('R²')
        else:
            plt.ylabel('MAE (Passengers)')
        plt.tight_layout()
        plt.show()
        plt.clf()
