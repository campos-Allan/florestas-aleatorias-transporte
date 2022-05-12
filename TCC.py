"""aplicacao de florestas aleatorias na base de dados
de transporte público de belo horizonte
"""
from datetime import date, timedelta
import time
from typing import List, Any
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def compilador_de_dados_passageiros(
        df_passageiro: pd.DataFrame,
        ano: int, mes: int,
        diretorio_arquivos: str) -> pd.DataFrame:
    """compilar planilhas e agrupar os dados de passageiros em um dataframe
    Args:
        df_passageiro (pd.DataFrame): agrupador dos dados
        ano (int): controle do ano pra abrir arquivos
        mes (int): controle do mês pra abrir arquivos
        diretorio_arquivos (str): caminho até a pasta das planilhas

    Returns:
        pd.DataFrame: df com dados das planilhas
    """
    while ano < 22:
        if mes < 10:
            caminho_planilhas = diretorio_arquivos + \
                '/mco-0'+str(mes)+'-20'+str(ano)+".csv"
        else:
            caminho_planilhas = diretorio_arquivos + \
                '/mco-'+str(mes)+'-20'+str(ano)+".csv"

        df_planilha = pd.DataFrame(pd.read_csv(caminho_planilhas, encoding='utf-8', sep=';',
                                               usecols=[' VIAGEM', ' LINHA',
                                                        ' CATRACA SAIDA', ' CATRACA CHEGADA']))
        # subtraindo valores de catraca saida da catraca chegada, calculando passageiros
        df_planilha['Passageiros'] = df_planilha.apply(lambda x: (
            100000-x[2]+x[3]) if (((x[3]-x[2]) < 0) & ((x[2]/100) >= 998)) else (x[3]-x[2]), axis=1)
        df_planilha = df_planilha.drop(
            [' CATRACA SAIDA', ' CATRACA CHEGADA'], axis=1)
        df_passageiro = pd.concat(
            [df_planilha, df_passageiro], ignore_index=True)

        print(str(mes)+'-20'+str(ano)+' OK')
        mes += 1
        if mes > 12:
            ano += 1
            mes = 1
    return df_passageiro


def grafico_passageiros(data: pd.DataFrame) -> None:
    """gerar gráfico scatter pra cada viagem

    Args:
        data (pd.DataFrame): df com viagens e passageiros
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)
    fig, axis = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(fig, left=True, bottom=True)
    sns.scatterplot(x=" VIAGEM", y="Passageiros",
                    palette="ch:r=-.2,d=.3_r",
                    sizes=(1, 8), linewidth=0,
                    data=data, ax=axis)
    plt.show()


def formatador_passageiros(df_passageiro: pd.DataFrame) -> pd.DataFrame:
    """formatação do dataframe de passageiros, gerando novos atributos

    Args:
        df_passageiro (pd.DataFrame): df pra ser formatado

    Returns:
        pd.DataFrame: df formatado
    """
    # renomeando colunas
    df_passageiro = df_passageiro.rename({'Valor': 'Passageiros'}, axis=1)
    df_passageiro = df_passageiro.rename({' VIAGEM': 'VIAGEM'}, axis=1)
    df_passageiro = df_passageiro.rename({' LINHA': 'LINHA'}, axis=1)

    # gerando novos atributos de data
    df_passageiro['Ano'] = df_passageiro['VIAGEM'].map(lambda x: x.year)
    df_passageiro['Mes'] = df_passageiro['VIAGEM'].map(lambda x: x.month)
    df_passageiro['Dia'] = df_passageiro['VIAGEM'].map(lambda x: x.day)
    df_passageiro['Dia da Semana'] = df_passageiro['VIAGEM'].map(
        lambda x: x.weekday())
    df_passageiro['Semana do ano'] = df_passageiro['VIAGEM'].apply(
        lambda x: x.weekofyear)

    # criando atributos do feriado, pré feriado e pós feriado
    df_passageiro['Pre Feriado'] = 0
    df_passageiro['Feriado'] = 0
    df_passageiro['Pos Feriado'] = 0
    feriados = [date(2016, 1, 1), date(2016, 2, 8), date(2016, 2, 9),
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
    df_passageiro['Feriado'] = df_passageiro['VIAGEM'].apply(
        lambda x: 1 if x in feriados else 0)
    dia = timedelta(1)
    pre_feriados = []
    pos_feriados = []
    for i in feriados:
        pre_feriados.append(i-dia)
        pos_feriados.append(i+dia)
    df_passageiro['Pre Feriado'] = df_passageiro['VIAGEM'].apply(
        lambda x: 1 if x in pre_feriados else 0)
    df_passageiro['Pos Feriado'] = df_passageiro['VIAGEM'].apply(
        lambda x: 1 if x in pos_feriados else 0)

    # gerando atributo da pandemia
    df_passageiro['Pandemia'] = df_passageiro["VIAGEM"].map(
        lambda x: 1 if x > date(2020, 3, 15) else 0)

    # substituindo os nomes das 319 linhas de ônibus por valores inteiro de 1 até 319
    df_passageiro['LINHA'] = df_passageiro['LINHA'].astype(str)
    df_passageiro['LINHA'] = df_passageiro['LINHA'].replace(
        list(df_passageiro['LINHA'].unique()), list(range(1, 320)))
    return df_passageiro


def compilador_de_dados_climaticos(
        df_chuva_temperatura: pd.DataFrame,
        ano: int, diretorio_arquivos: str) -> pd.DataFrame:
    """agrupar os dados climáticos em um dataframe

    Args:
        df_chuva_temperatura (pd.DataFrame): dataframe agrupador
        ano (int): controle do ano pra abrir arquivos
        diretorio_arquivos (str): caminho até a pasta das planilhas

    Returns:
        pd.DataFrame: dataframe com dados agrupados
    """

    # nas planilhas de dados climáticos do INMET, antes de se rodar o código,
    # excluiu-se as 8 primeiras linhas e todas colunas,
    # com exceção de Data, Hora, Temperatura do Ar e Precipitação Total
    while ano < 22:
        caminho = diretorio_arquivos+'/mg'+str(ano)+'.csv'
        df_planilha = pd.DataFrame(pd.read_csv(
            caminho, encoding='utf-8', sep=';'))
        df_chuva_temperatura = pd.concat(
            [df_planilha, df_chuva_temperatura], ignore_index=True)
        ano += 1
    return df_chuva_temperatura


def formatador_clima(df_chuva_temperatura: pd.DataFrame) -> pd.DataFrame:
    """formatação do dataFrame climático, substituindo valores anômalos

    Args:
        df_clima (pd.DataFrame): dataframe pra ser formatado

    Returns:
        pd.DataFrame: df formatado
    """
    df_chuva_temperatura['Chuva'] = df_chuva_temperatura['Chuva'].str.replace(
        ",", ".")
    df_chuva_temperatura['Temp'] = df_chuva_temperatura['Temp'].str.replace(
        ",", ".")

    # substituindo valores '-9999' por valores nulos
    df_chuva_temperatura['Chuva'] = df_chuva_temperatura['Chuva'].replace(
        "-9999", np.nan)
    df_chuva_temperatura['Temp'] = df_chuva_temperatura['Temp'].replace(
        "-9999", np.nan)

    df_chuva_temperatura['Chuva'] = df_chuva_temperatura['Chuva'].astype(float)
    df_chuva_temperatura['Temp'] = df_chuva_temperatura['Temp'].astype(float)

    # preenchendo valores nulos de chuva por 0
    # e valores nulos de temperatura pela média
    df_chuva_temperatura['Chuva'] = df_chuva_temperatura['Chuva'].fillna(0)
    df_chuva_temperatura['Temp'] = df_chuva_temperatura['Temp'].fillna(
        df_chuva_temperatura['Temp'].mean())

    # agrupando medições horárias por dia, e gerando novas colunas,
    # do somatório total de chuva, além de valores médios,
    # máximos e mínimos de temperatura
    df_chuva_temperatura = df_chuva_temperatura.groupby('Data', axis=0, as_index=False).agg(
        {'Chuva': 'sum', 'Temp': ['mean', 'max', 'min']})
    df_chuva_temperatura.columns = df_chuva_temperatura.columns.droplevel(0)

    # mexendo na coluna de data (VIAGEM) para criar compatibilidade
    # com o dataframe de passageiros,
    # e juntar os dois dataframes a partir dela
    df_chuva_temperatura[''] = pd.to_datetime(
        df_chuva_temperatura[''], dayfirst=True)
    df_chuva_temperatura = df_chuva_temperatura.sort_values(by='')
    df_chuva_temperatura = df_chuva_temperatura.rename(columns={'': 'VIAGEM'})

    return df_chuva_temperatura


def timesplit(
        time_split: TimeSeriesSplit,
        atributos_x: pd.DataFrame, passageiros_y: pd.Series,
        n_estimators: int, max_depth: int) -> List[List]:
    """iterações que percorrem os dados,
    dividindo conjuntos de treino e teste

    Args:
        time_split (TimeSeriesSplit): modelo janela crescente com validação adiante
        atributos_x (pd.DataFrame): atributos do modelo
        passageiros_y (pd.Series): variável passageiros
        n_estimators (int): quantidade de árvores na floresta
        max_depth (int): profundidade máxima da árvore

    Returns:
        List[List]: resultados a cada iteração
    """
    reg = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    resultados = []
    for fold, (train_index, test_index) in enumerate(time_split.split(atributos_x)):
        start = time.time()  # Início do cálculo do tempo decorrido

        # índices de treino e teste de acordo com a
        # iteração da janela crescente com validação adiante
        print(f"Fold: {fold}")
        print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
        print("\n")

        # divisão dos dados entre treino e teste,
        # nos atributos (X) e variável 'passageiros' (Y)
        x_train, x_test = atributos_x.iloc[train_index], atributos_x.iloc[test_index]
        y_train, y_test = passageiros_y.iloc[train_index], passageiros_y.iloc[test_index]

        reg.fit(x_train, y_train)  # fit com valores de treino
        acc = round(reg.score(x_test, y_test) * 100, 2)  # R²
        y_pred = reg.predict(x_test)  # prevendo valores de teste
        erro_ab = mean_absolute_error(
            y_test, y_pred)  # MAE com valores preditos

        end = time.time()  # final do cálculo do tempo decorrido

        resultados.append([acc, erro_ab, train_index[-1], (end-start)])
        print("R2 do Regressor de Floresta Aleatória:", acc, "%")
        # criação dos gráficos tipo scatter da distribuição das previsões
        grafico_distribuicao_previsoes(
            y_test, y_pred, x_train, max_depth, n_estimators)

    return resultados


def florestas_aleatorias(
        time_split: TimeSeriesSplit,
        instancia: str,
        atributos_x: pd.DataFrame, passageiros_y: pd.Series) -> pd.DataFrame:
    """ modelos de floresta aleatória com diferentes hiperparâmetros
    sendo iterados na função anterior

    Args:
        time_split (TimeSeriesSplit): modelo janela crescente com validação adiante
        instancia (str): pandemia, pré pandemia ou conjunto completo
        atributos_x (pd.DataFrame): atributos do modelo
        passageiros_y (pd.Series): variável passageiros

    Returns:
        pd.DataFrame: resultados das métricas de desempenho
    """
    # parâmetros de quantidade de árvores
    nos = [2, 10, 20, 50, 100]
    # parâmetros de profundidade máxima da árvore
    profund = [5, 10, 20, 50, 100]
    resultados = pd.DataFrame()
    for i in nos:
        for j in profund:
            resultados[instancia+' DATAFRAME, '+'A' +
                       str(i)+' N'+str(j)] = timesplit(time_split, atributos_x, passageiros_y, i, j)
    return resultados


def grafico_distribuicao_previsoes(
        y_test: Any, y_pred: Any,
        x_train: pd.Series,
        max_depth: int, n_estimators: int) -> None:
    """geração do gráfico que mostra os valores das
    previsões frente aos valores reais
    """
    df_fa = pd.DataFrame({'Valor Real': y_test, 'Previsão': y_pred})
    plt.scatter(y_test, y_pred)  # plot scatter
    # linha vermelha com regressão dos pontos,
    # a linha verde do x=y foi traçada no GIMP
    sns.regplot(x="Valor Real", y="Previsão",
                data=df_fa, color='red', marker="")

    # salvando gráficos de acordo com a divisão de dados trabalhada
    if len(X) > 410000:
        plt.savefig('TODO DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(x_train))+'.png')
    elif len(X) < 200000:
        plt.savefig('PAN DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(x_train))+'.png')
    else:
        plt.savefig('PRE DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(x_train))+'.png')
    plt.clf()


def graficos_resultados(resultados: List[List]) -> None:
    """geração dos gráficos das métricas de desempenho
    do modelo A100 N100


    Args:
        resultados (List[List]): resultados das métricas
    """
    # dividindo as métricas a partir dos resultados
    resultados_r2 = resultados.applymap(lambda x: x[0])
    resultados_mae = resultados.applymap(lambda x: x[1])
    resultados_tempo = resultados.applymap(lambda x: x[3])

    # plotando com eixo secundário os valores de MAE e R²
    # de acordo com iterações da janela para o modelo A100N100
    fig, ax1 = plt.subplots()
    fig.subtitle('A100 N100')
    ax1.plot(resultados_r2.iloc[:, -1], color="red", marker="o")
    ax1.set_xlabel("Iterações da Janela", fontsize=14)
    ax1.set_ylabel("R²", color="red", fontsize=14)
    ax2 = ax1.twinx()
    ax2.plot(resultados_mae.iloc[:, -1], color="blue", marker="o")
    ax2.set_ylabel("Erro Médio Absoluto (Passageiros)",
                   color="blue", fontsize=14)
    plt.tight_layout()
    plt.show()

    # chamando a próxima função para plotar métricas
    # de desempenho para todos modelos de Floresta Aleatória
    plotador_modelos(resultados_r2, 'box_r2')
    plotador_modelos(resultados_mae, 'box_mae')
    plotador_modelos(resultados_tempo, 'plt')


def plotador_modelos(resultados: Any, tipo: str) -> None:
    """gráficos dos desempenhos de MAE e R² de cada
    modelo de Floresta Aleatória em boxplot, além do gráfico
    de tempo

    Args:
        resultados (Any): resultados dos modelos
        tipo (str): tipo de gráfico, sendo boxplot ou plot do tempo
    """
    resultados.columns = ['A2 N5', 'A2 N10', 'A2 N20', 'A2 N50', 'A2 N100',
                          'A10 N5', 'A10 N10', 'A10 N20', 'A10 N50',
                          'A10 N100', 'A20 N5', 'A20 N10', 'A20 N20',
                          'A20 N50', 'A20 N100', 'A50 N5', 'A50 N10',
                          'A50 N20', 'A50 N50', 'A50 N100', 'A100 N5',
                          'A100 N10', 'A100 N20', 'A100 N50', 'A100 N100']
    sns.set_style("whitegrid")
    if tipo == 'plt':
        # plot do tempo decorrido por cada modelo de Floresta Aleatória
        resultados = resultados.sum()
        resultados.plot(fontsize=14)
        plt.xlabel('Modelos de Floresta Aleatória')
        plt.ylabel('Segundos')
        plt.tight_layout()
        plt.show()
        plt.clf()
    else:
        # boxplot dos valores de R² e MAE para cada modelo de Floresta Aleatória
        plt.boxplot(resultados)
        plt.xticks(ticks=range(len(resultados.columns)),
                   labels=resultados.columns, rotation=45)
        plt.xlabel('Modelos de Floresta Aleatória')
        if tipo == 'box_r2':
            plt.ylabel('R²')
        else:
            plt.ylabel('Erro Médio Absoluto (Passageiros)')
        plt.tight_layout()
        plt.show()
        plt.clf()


# compilação dos dados de passageiros
df_passageiros = pd.DataFrame()
df_passageiros = compilador_de_dados_passageiros(
    df_passageiros, 16, 1, 'C:/Users/unkno/Downloads/mco')

df_passageiros[' VIAGEM'] = pd.to_datetime(
    df_passageiros[' VIAGEM'], dayfirst=True)
df_passageiros = df_passageiros.sort_values(by=' VIAGEM')

# retirando valores anômalos com mais de 500 passageiros por viagem,
# além de valores zerados e negativos de passageiros
maiores = df_passageiros[df_passageiros.Passageiros > 500]
df_passageiros = df_passageiros[df_passageiros.Passageiros < 500]
zeros = df_passageiros[df_passageiros.Passageiros == 0]
df_passageiros = df_passageiros[df_passageiros.Passageiros != 0]
negativos = df_passageiros[df_passageiros.Passageiros < 0]
df_passageiros = df_passageiros[df_passageiros.Passageiros > 0]

# gerando gráfico de todas as viagens,
# antes e depois destas serem agrupadas por dia
grafico_passageiros(df_passageiros)
df_passageiros = df_passageiros.groupby(
    [' VIAGEM', ' LINHA'], axis=0, as_index=False).sum()
grafico_passageiros(df_passageiros)

# formatar o dataframe de passageiros
df_passageiros = formatador_passageiros(df_passageiros)

# compilação dos dados climáticos e formatar
df_clima = pd.DataFrame()
df_clima = compilador_de_dados_climaticos(
    df_clima, 16, 'C:/Users/unkno/Desktop/resultados')
df_clima = formatador_clima(df_clima)

# juntado os dados de passageiros com dados climáticos
df_final = pd.merge(df_passageiros, df_clima, on='VIAGEM')
df_final = df_final.rename(columns={
                           'sum': 'Chuva',
                           'mean': 'Temperatura media',
                           'max': 'Temperatura Max',
                           'min': 'Temperatura Min'})
df_final = df_final.drop("VIAGEM", axis=1)

# separando todos atributos de clima e calendário(X),
# da variável do número de passageiros (Y),
# que será predita no método de Florestas Aleatórias
X = df_final.copy()
y = X.pop("Passageiros")

# primeiro experimento sendo realizados no período completo de dados
tscv = TimeSeriesSplit(n_splits=24)  # +- 3 meses
Resultados_periodocompleto = pd.DataFrame()
Resultados_periodocompleto = florestas_aleatorias(tscv, 'TODO', X, y)

# Divindo o conjunto de dados para separar
# o período sem os anos pandêmicos (2016-2019)
# e o período com os anos pandêmicos (2020-2021)
X_pan = X[400038:]
y_pan = y[400038:]

X_pre = X[:400038]
y_pre = y[:400038]

# Segundo experimento sendo realizados
# no período de dados sem os anos pandêmicos
tscv = TimeSeriesSplit(n_splits=16)  # +- 3 meses
Resultados_pre_pandemia = pd.DataFrame()
Resultados_pre_pandemia = florestas_aleatorias(tscv, 'PRE', X_pre, y_pre)

# Terceiro experimento sendo realizados
# no período de dados com os anos pandêmicos
tscv = TimeSeriesSplit(n_splits=8)  # +- 3 meses
Resultados_pan = pd.DataFrame()
Resultados_pan = florestas_aleatorias(tscv, 'PAN', X_pan, y_pan)

# Gerando gráficos finais das métricas de desempenho
# para cada conjunto de experimento feito
graficos_resultados(Resultados_periodocompleto)
graficos_resultados(Resultados_pre_pandemia)
graficos_resultados(Resultados_pan)
