'''graphic generator
'''

from typing import Any, List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


def passenger_alltrips_graphic(data: pd.DataFrame) -> None:
    '''scatter plot all trips, before and after groupby per data & bus line

    Args:
        data (pd.DataFrame): dataframe with trips
    '''
    sns.set_theme(style='whitegrid')
    # sns.set(font_scale=2)
    fig, axis = plt.subplots(figsize=(8, 8))
    sns.despine(fig, left=True, bottom=True)
    sns.scatterplot(x='trip', y='passenger',
                    palette='ch:r=-.2,d=.3_r',
                    sizes=(1, 8), linewidth=0,
                    data=data, ax=axis)
    plt.tight_layout()
    plt.savefig('all_graphics/alltrips'+str(len(data))+'.png')
    plt.close(fig)


def realvalue_vs_prediction_graphic(
        y_test: Any, y_pred: Any,
        x_train: pd.Series, x_test: pd.Series,
        acc: float, erro_ab: float, x_features: pd.DataFrame) -> None:
    '''scatter plot comparing predictions vs real values
    for each iteration of each dataset experiment
    '''
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(y_test, y_pred)
    regression = linregress(y_test, y_pred)
    axis.plot(y_test, y_test*regression.slope +
              regression.intercept, 'r', label='fitted line')
    axis.plot(y_test, y_test, 'g', label='perfect predictions')
    axis.legend()
    axis.set_title('Model A100N100 - Training: 0 - ' + str(len(x_train)-1)+' Testing: '+str(
        len(x_train)) + ' - '+str(len(x_test)+len(x_train)-1)+' / R²: '+str(acc)+' / MAE: '+str(
            round(erro_ab, 1)))
    axis.set_xlabel('Real Value')
    axis.set_ylabel('Prediction')
    plt.tight_layout()
    # saving graphics of each experiment
    if len(x_features) > 410000:
        plt.savefig(
            'all_graphics/realvalue_vs_prediction/TODO DATAFRAME A100 N100 '+'-Training '+str(
                round(100*len(x_train)/len(x_features), 1)) + ' percent of data'+'.png')
    elif len(x_features) < 200000:
        plt.savefig(
            'all_graphics/realvalue_vs_prediction/PAN DATAFRAME A100 N100 '+'-Training '+str(
                round(100*len(x_train)/len(x_features), 1)) + ' percent of data'+'.png')
    else:
        plt.savefig(
            'all_graphics/realvalue_vs_prediction/PRE DATAFRAME A100 N100'+'-Training '+str(
                round(100*len(x_train)/len(x_features), 1)) + ' percent of data'+'.png')
    plt.close(fig)


def model_graphics(results: List[List]) -> None:
    '''generating graphics of peformance metrics from
    model with hyperparameters tuned to max


    Args:
        resultados (List[List]): perfomance metrics
    '''
    results_r2 = results.applymap(lambda x: x[0])
    results_mae = results.applymap(lambda x: x[1])
    train_r2 = results.applymap(lambda x: x[2])
    train_mae = results.applymap(lambda x: x[3])
    results_time = results.applymap(lambda x: x[5])

    fig, ax1 = plt.subplots(figsize=(8, 8))

    fig.suptitle('A100 N100')  # A= Nodes; N= Max Depth
    ax1.plot(results_r2.iloc[:, -1], color='red', marker='o', label='R² test')
    ax1.plot(train_r2.iloc[:, -1], color='black', marker='o', label='R² train')
    ax1.set_xlabel('Window Iterations', fontsize=14)
    ax1.set_ylabel('R²', color='red', fontsize=14)
    ax1.grid()
    ax1.legend(loc='center left')
    ax2 = ax1.twinx()
    ax2.plot(results_mae.iloc[:, -1], color='blue',
             marker='o', label='MAE test')
    ax2.plot(train_mae.iloc[:, -1], color='green',
             marker='o', label='MAE train')
    ax2.set_ylabel('MAE (Passengers)',
                   color='blue', fontsize=14)
    ax2.legend(loc='center right')
    plt.tight_layout()
    plt.savefig('all_graphics/model/A100 N100 model' +
                str(len(results_r2))+'.png')
    plt.close(fig)

    # boxtplotting the result from every random forest model with different
    # hyperparameters
    boxplotter(results_r2, 'box_r2')
    boxplotter(results_mae, 'box_mae')
    boxplotter(results_time, 'plt')


def boxplotter(results: Any, graphic_type: str) -> None:
    '''boxtplotting the result from every random forest model
    with different hyperparameters, plus elapsed time

    Args:
        resultados (Any): perfomance metrics
        graphic_type (str): type of graphic: random forest boxplot or time plot
    '''
    results.columns = ['A2 N5', 'A2 N10', 'A2 N20', 'A2 N50', 'A2 N100',
                       'A10 N5', 'A10 N10', 'A10 N20', 'A10 N50',
                       'A10 N100', 'A20 N5', 'A20 N10', 'A20 N20',
                       'A20 N50', 'A20 N100', 'A50 N5', 'A50 N10',
                       'A50 N20', 'A50 N50', 'A50 N100', 'A100 N5',
                       'A100 N10', 'A100 N20', 'A100 N50', 'A100 N100']
    sns.set_style('whitegrid')
    if graphic_type == 'plt':
        # elapsed time graphic in each random forest model
        fig, axis = plt.subplots(figsize=(12, 8))
        len_id = len(results)
        graph_ticks = results.columns
        results = results.sum()
        results.plot(ax=axis, fontsize=14)
        axis.set_title('Elapsed Time')
        axis.set_xticks(ticks=range(len(graph_ticks)),
                        labels=graph_ticks, rotation=45)
        axis.set_ylabel('Seconds')
        axis.set_xlabel('Random Forest Models')
        plt.tight_layout()
        plt.savefig('all_graphics/boxplot_and_timeelapsed_time' +
                    str(len_id)+'.png')
    else:
        # R² and MAE boxplot in each random forest model
        fig, axis = plt.subplots(figsize=(16, 8))
        sns.boxplot(data=results, ax=axis)
        axis.grid(True)
        axis.set_xticks(ticks=range(len(results.columns)),
                        labels=results.columns, rotation=45)
        axis.set_xlabel('Random Forest Models')
        if graphic_type == 'box_r2':
            axis.set_ylabel('R²')
            axis.set_ylim(0, 100)
        else:
            axis.set_ylabel('MAE (Passengers)')
        axis.set_title('All random forest models in Boxplot')
        plt.tight_layout()
        plt.savefig('all_graphics/boxplot_and_time'+graphic_type +
                    ' boxplot'+str(len(results))+'.png')
    plt.close(fig)
