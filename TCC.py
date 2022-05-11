# IMPORTANDO BIBLIOTECAS
from datetime import date, timedelta
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def compilador_de_dados_passageiros(df_passageiro: pd.DataFrame, ano: int, mes: int, diretorio_arquivos: str) -> pd.DataFrame:
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

        df = pd.DataFrame(pd.read_csv(caminho_planilhas, encoding='utf-8', sep=';',
                                      usecols=[' VIAGEM', ' LINHA', ' CATRACA SAIDA', ' CATRACA CHEGADA']))
        # Subtraindo valores de catraca saida da catraca chegada, calculando passageiros
        df['Passageiros'] = df.apply(lambda x: (
            100000-x[2]+x[3]) if (((x[3]-x[2]) < 0) & ((x[2]/100) >= 998)) else (x[3]-x[2]), axis=1)
        df = df.drop(
            [' CATRACA SAIDA', ' CATRACA CHEGADA'], axis=1)
        df_passageiro = pd.concat(
            [df, df_passageiro], ignore_index=True)

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
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(x=" VIAGEM", y="Passageiros",
                    palette="ch:r=-.2,d=.3_r",
                    sizes=(1, 8), linewidth=0,
                    data=data, ax=ax)
    plt.show()


def formatador_passageiros(df_passageiros: pd.DataFrame) -> pd.DataFrame:
    """formatação do dataframe de passageiros, gerando novos atributos

    Args:
        df_passageiros (pd.DataFrame): df pra ser formatado

    Returns:
        pd.DataFrame: df formatado
    """
    # Renomeando colunas
    df_passageiros = df_passageiros.rename({'Valor': 'Passageiros'}, axis=1)
    df_passageiros = df_passageiros.rename({' VIAGEM': 'VIAGEM'}, axis=1)
    df_passageiros = df_passageiros.rename({' LINHA': 'LINHA'}, axis=1)

    # Gerando novos atributos de data
    df_passageiros['Ano'] = df_passageiros['VIAGEM'].map(lambda x: x.year)
    df_passageiros['Mes'] = df_passageiros['VIAGEM'].map(lambda x: x.month)
    df_passageiros['Dia'] = df_passageiros['VIAGEM'].map(lambda x: x.day)
    df_passageiros['Dia da Semana'] = df_passageiros['VIAGEM'].map(
        lambda x: x.weekday())
    df_passageiros['Semana do ano'] = df_passageiros['VIAGEM'].apply(
        lambda x: x.weekofyear)

    # Criando atributos do feriado, pré feriado e pós feriado
    df_passageiros['Pre Feriado'] = 0
    df_passageiros['Feriado'] = 0
    df_passageiros['Pos Feriado'] = 0
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
    df_passageiros['Feriado'] = df_passageiros['VIAGEM'].apply(
        lambda x: 1 if x in feriados else 0)
    dia = timedelta(1)
    pre_feriados = []
    pos_feriados = []
    for i in feriados:
        pre_feriados.append(i-dia)
        pos_feriados.append(i+dia)
    df_passageiros['Pre Feriado'] = df_passageiros['VIAGEM'].apply(
        lambda x: 1 if x in pre_feriados else 0)
    df_passageiros['Pos Feriado'] = df_passageiros['VIAGEM'].apply(
        lambda x: 1 if x in pos_feriados else 0)

    # Gerando atributo da pandemia
    df_passageiros['Pandemia'] = df_passageiros["VIAGEM"].map(
        lambda x: 1 if x > date(2020, 3, 15) else 0)

    # Substituindo os nomes das 319 linhas de ônibus por valores inteiro de 1 até 319
    df_passageiros['LINHA'] = df_passageiros['LINHA'].astype(str)
    df_passageiros['LINHA'] = df_passageiros['LINHA'].replace(
        list(df_passageiros['LINHA'].unique()), list(range(1, 320)))
    return df_passageiros


def compilador_de_dados_climaticos(df_clima: pd.DataFrame, ano: int, diretorio_arquivos: str) -> pd.DataFrame:
    """agrupar os dados climáticos em um dataframe

    Args:
        df_clima (pd.DataFrame): dataframe agrupador
        ano (int): controle do ano pra abrir arquivos
        diretorio_arquivos (str): caminho até a pasta das planilhas

    Returns:
        pd.DataFrame: dataframe com dados agrupados
    """

    # Nas planilhas de dados climáticos do INMET, antes de se rodar o código,
    # excluiu-se as 8 primeiras linhas e todas colunas,
    # com exceção de Data, Hora, Temperatura do Ar e Precipitação Total
    while ano < 22:
        caminho = diretorio_arquivos+'/mg'+str(ano)+'.csv'
        df = pd.DataFrame(pd.read_csv(caminho, encoding='utf-8', sep=';'))
        df_clima = pd.concat([df, df_clima], ignore_index=True)
        ano += 1
    return df_clima


def formatador_clima(df_clima: pd.DataFrame) -> pd.DataFrame:
    """formatação do dataFrame climático, substituindo valores anômalos

    Args:
        df_clima (pd.DataFrame): dataframe pra ser formatado

    Returns:
        pd.DataFrame: df formatado
    """
    df_clima['Chuva'] = df_clima['Chuva'].str.replace(",", ".")
    df_clima['Temp'] = df_clima['Temp'].str.replace(",", ".")

    # Substituindo valores '-9999' por valores nulos
    df_clima['Chuva'] = df_clima['Chuva'].replace("-9999", np.nan)
    df_clima['Temp'] = df_clima['Temp'].replace("-9999", np.nan)

    df_clima['Chuva'] = df_clima['Chuva'].astype(float)
    df_clima['Temp'] = df_clima['Temp'].astype(float)

    # Preenchendo valores nulos de chuva por 0
    # e valores nulos de temperatura pela média
    df_clima['Chuva'] = df_clima['Chuva'].fillna(0)
    df_clima['Temp'] = df_clima['Temp'].fillna(df_clima['Temp'].mean())

    # Agrupando medições horárias por dia, e gerando novas colunas,
    # do somatório total de chuva, além de valores médios,
    # máximos e mínimos de temperatura
    df_clima = df_clima.groupby('Data', axis=0, as_index=False).agg(
        {'Chuva': 'sum', 'Temp': ['mean', 'max', 'min']})
    df_clima.columns = df_clima.columns.droplevel(0)

    # Mexendo na coluna de data (VIAGEM) para criar compatibilidade com o DataFrame de passageiros, e juntar os dois DataFrames a partir dela
    df_clima[''] = pd.to_datetime(df_clima[''], dayfirst=True)
    df_clima = df_clima.sort_values(by='')
    df_clima = df_clima.rename(columns={'': 'VIAGEM'})

    return df_clima

# 6-> Iterações que percorrem os dados, dividindo conjuntos de treino e teste, seguindo o modelo janela crescente com validação adiante (timesplit)


def timesplit(tscv, X, y, n_estimators, max_depth):
    # Modelo de Floresta Aleatória de Regressão
    reg = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    resultados = []
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        start = time.time()  # Início do cálculo do tempo decorrido

        # Índices de treino e teste de acordo com a iteração da janela crescente com validação adiante
        print("Fold: {}".format(fold))
        print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
        print("\n")

        # Divisão dos dados entre treino e teste, nos atributos (X) e variável 'Passageiros' (Y)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        reg.fit(X_train, y_train)  # Fit com valores de treino
        # R² com valores de teste
        acc = round(reg.score(X_test, y_test) * 100, 2)
        y_pred = reg.predict(X_test)  # Prevendo valores de teste
        erro_ab = mean_absolute_error(
            y_test, y_pred)  # MAE com valores preditos

        end = time.time()  # Final do cálculo do tempo decorrido

        resultados.append([acc, erro_ab, train_index[-1], (end-start)])
        print("R2 do Regressor de Floresta Aleatória:", acc, "%")
        # Criação dos gráficos tipo scatter da distribuição das previsões frente aos valores reais
        grafico_distribuição_previsoes(
            y_test, y_pred, X_train, max_depth, n_estimators)

    return resultados

# 7-> Modelos de Floresta Aleatória com diferentes hiperparâmetros sendo iterados na função 6, e reunindo os resultados em um DataFrame (florestas_aleatorias)


def florestas_aleatorias(tscv, instancia, X, y):
    # Parâmetros de quantidade de árvores
    nos = [2, 10, 20, 50, 100]
    # Paâmetros de profundidade máxima da árvore
    profund = [5, 10, 20, 50, 100]
    Resultados = pd.DataFrame()
    for i in nos:
        for j in profund:
            # Salvando no DataFrame de resultados da função acima de acordo com os hiperparâmetros usados por cada modelo
            Resultados[instancia+' DATAFRAME, '+'A' +
                       str(i)+' N'+str(j)] = timesplit(tscv, X, y, i, j)
    return Resultados

# 8-> Geração do gráfico que mostra os valores das previsões frente aos valores reais, e sua distribuição num gráfico tipo scatter (grafico_distribuição_previsões)


def grafico_distribuição_previsoes(y_test, y_pred, X_train, max_depth, n_estimators):
    df_fa = pd.DataFrame({'Valor Real': y_test, 'Previsão': y_pred})
    plt.scatter(y_test, y_pred)  # Plot Scatter
    # Linha vermelha com regressão dos pontos, a linha verde do x=y foi traçada no GIMP
    ax = sns.regplot(x="Valor Real", y="Previsão",
                     data=df_fa, color='red', marker="")

    # Salvando gráficos de acordo com a divisão de dados trabalhada
    if len(X) > 410000:
        plt.savefig('TODO DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(X_train))+'.png')
    elif len(X) < 200000:
        plt.savefig('PAN DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(X_train))+'.png')
    else:
        plt.savefig('PRE DATAFRAME A'+str(n_estimators)+" N" +
                    str(max_depth)+' '+str(len(X_train))+'.png')
    plt.clf()

# 9-> Geração dos gráficos das métricas de desempenho do modelo A100 N100 no decorrer das iterações da função 6, além de chamar a função seguinte para mais gráficos (graficos_resultados)


def graficos_resultados(Resultados):
    # Dividindo as métricas a partir do DataFrame dos resultados
    Resultados_R2 = Resultados.applymap(lambda x: x[0])
    Resultados_mae = Resultados.applymap(lambda x: x[1])
    Resultados_tempo = Resultados.applymap(lambda x: x[3])

    # Plotando com eixo secundário os valores de MAE e R² de acordo com iterações da janela para o modelo A100N100
    fig, ax = plt.subplots()
    ax.plot(Resultados_R2.iloc[:, -1], color="red", marker="o")
    ax.set_xlabel("Iterações da Janela", fontsize=14)
    ax.set_ylabel("R²", color="red", fontsize=14)
    ax2 = ax.twinx()
    ax2.plot(Resultados_mae.iloc[:, -1], color="blue", marker="o")
    ax2.set_ylabel("Erro Médio Absoluto (Passageiros)",
                   color="blue", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Chamando a próxima função para plotar métricas de desempenho para todos modelos de Floresta Aleatória
    plotador_modelos(Resultados_R2, 'box_r2')
    plotador_modelos(Resultados_mae, 'box_mae')
    plotador_modelos(Resultados_tempo, 'plt')

# 10-> Geração dos gráficos dos desempenhos de MAE e R² de cada modelo de Floresta Aleatória em um boxplot, além do gráfico de tempo decorrido pelos modelos (plotador_modelos)


def plotador_modelos(db, tipo):
    db.columns = ['A2 N5', 'A2 N10', 'A2 N20', 'A2 N50', 'A2 N100', 'A10 N5', 'A10 N10', 'A10 N20', 'A10 N50', 'A10 N100', 'A20 N5', 'A20 N10',
                  'A20 N20', 'A20 N50', 'A20 N100', 'A50 N5', 'A50 N10', 'A50 N20', 'A50 N50', 'A50 N100', 'A100 N5', 'A100 N10', 'A100 N20', 'A100 N50', 'A100 N100']
    sns.set_style("whitegrid")
    if tipo == 'plt':
        # Plot do tempo decorrido por cada modelo de Floresta Aleatória
        db = db.sum()
        db.plot(fontsize=14)
        plt.xlabel('Modelos de Floresta Aleatória')
        plt.ylabel('Segundos')
        plt.tight_layout()
        plt.show()
        plt.clf()
    else:
        # Boxplot dos valores de R² e MAE para cada modelo de Floresta Aleatória
        plt.boxplot(db)
        plt.xticks(ticks=range(len(db.columns)),
                   labels=db.columns, rotation=45)
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
                           'sum': 'Chuva', 'mean': 'Temperatura media', 'max': 'Temperatura Max', 'min': 'Temperatura Min'})
df_final = df_final.drop("VIAGEM", axis=1)

# Separando todos atributos de clima e calendário(X), da variável do número de passageiros (Y), que será predita no método de Florestas Aleatórias
X = df_final.copy()
y = X.pop("Passageiros")

# Primeiro experimento sendo realizados no período completo de dados
# tscv = TimeSeriesSplit(n_splits=24)  # +- 3 meses
#Resultados_periodocompleto = pd.DataFrame()
#Resultados_periodocompleto = florestas_aleatorias(tscv, 'TODO', X, y)

# Divindo o conjunto de dados para separar o período sem os anos pandêmicos (2016-2019) e o período com os anos pandêmicos (2020-2021)
X_pan = X[400038:]
y_pan = y[400038:]

X_pre = X[:400038]
y_pre = y[:400038]

# Segundo experimento sendo realizados no período de dados sem os anos pandêmicos
# tscv = TimeSeriesSplit(n_splits=16)  # +- 3 meses
#Resultados_pre_pandemia = pd.DataFrame()
#Resultados_pre_pandemia = florestas_aleatorias(tscv, 'PRE', X_pre, y_pre)

# Terceiro experimento sendo realizados no período de dados com os anos pandêmicos
tscv = TimeSeriesSplit(n_splits=8)  # +- 3 meses
Resultados_pan = pd.DataFrame()
Resultados_pan = florestas_aleatorias(tscv, 'PAN', X_pan, y_pan)

# Gerando gráficos finais das métricas de desempenho para cada conjunto de experimento feito
# graficos_resultados(Resultados_periodocompleto)
# graficos_resultados(Resultados_pre_pandemia)
graficos_resultados(Resultados_pan)
