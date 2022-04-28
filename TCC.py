#IMPORTANDO BIBLIOTECAS
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

#FUNÇÕES
# 1-> Compilar planilhas e agrupar os dados de passageiros em um DataFrame(compilador_de_dados_passageiros)
def compilador_de_dados_passageiros(Passageiros,ano,mes,diretorio_arquivos):
    while ano<22:
        if mes<10:
            caminho_planilhas=diretorio_arquivos+'/mco-0'+str(mes)+'-20'+str(ano)+".csv"
        else:
            caminho_planilhas=diretorio_arquivos+'/mco-'+str(mes)+'-20'+str(ano)+".csv"
            
        df=pd.DataFrame(pd.read_csv(caminho_planilhas,encoding='utf-8',sep=';',usecols=[' VIAGEM',' LINHA',' CATRACA SAIDA',' CATRACA CHEGADA']))
        #Subtraindo valores de catraca saida da catraca chegada, para calcular o número de passageiros transportados
        df['Passageiros']=df.apply(lambda x:(100000-x[2]+x[3]) if (((x[3]-x[2])<0) & ((x[2]/100)>=998)) else (x[3]-x[2]) ,axis=1)
        df=df.drop([' CATRACA SAIDA',' CATRACA CHEGADA'],axis=1)
        Passageiros=pd.concat([df,Passageiros], ignore_index = True)

        print(str(mes)+'-20'+str(ano)+' OK')
        mes+=1
        if mes>12:
            ano+=1
            mes=1
    return Passageiros

# 2-> Gerar gráfico onde cada ponto representa uma viagem, com caracaterística de data e quantidade de passageiros (grafico_passageiros)
def grafico_passageiros(data):
    sns.set_theme(style="whitegrid")
    sns.set(font_scale = 2)
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(x=" VIAGEM", y="Passageiros",
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=data, ax=ax)
    plt.show()

# 3-> Formatação do DataFrame de passageiros, gerando novos atributos em relação a data das viagens, feriados e pandemia (formatador_passageiros)
def formatador_passageiros(db):
    db=db.rename({'Valor':'Passageiros'}, axis=1)
    
    db['Ano']=db[' VIAGEM'].map(lambda x: x.year)
    db['Mes']=db[' VIAGEM'].map(lambda x: x.month)
    db['Dia']=db[' VIAGEM'].map(lambda x: x.day)
    db['Dia da Semana']=db[' VIAGEM'].map(lambda x:x.weekday())
    db['Semana do ano']=db[' VIAGEM'].apply(lambda x: x.weekofyear)

    #Criando atributos do feriado, pré feriado e pós feriado
    db['Pre Feriado']=0
    db['Feriado']=0
    db['Pos Feriado']=0
    feriados=[date(2016,1,1),date(2016,2,8),date(2016,2,9),date(2016,2,10),date(2016,3,25),date(2016,4,21),date(2016,5,1),date(2016,5,26),date(2016,8,15),date(2016,9,7),date(2016,10,12),date(2016,11,2),date(2016,11,15),date(2016,12,8),date(2016,12,25),date(2017,1,1),date(2017,2,27),date(2017,2,28),date(2017,3,1),date(2017,4,14),date(2017,4,21),date(2017,5,1),date(2017,6,15),date(2017,8,15),date(2017,9,7),date(2017,10,12),date(2017,11,2),date(2017,11,15),date(2017,12,8),date(2017,12,25),date(2018,1,1),date(2018,2,12),date(2018,2,13),date(2018,2,14),date(2018,3,30),date(2018,4,21),date(2018,5,1),date(2018,5,31),date(2018,8,15),date(2018,9,7),date(2018,10,12),date(2018,11,2),date(2018,11,15),date(2018,12,8),date(2018,12,25),date(2019,1,1),date(2019,3,4),date(2019,3,5),date(2019,3,6),date(2019,4,19),date(2019,4,21),date(2019,5,1),date(2019,6,20),date(2019,8,15),date(2019,9,7),date(2019,10,12),date(2019,11,2),date(2019,11,15),date(2019,12,8),date(2019,12,25),date(2020,1,1),date(2020,2,24),date(2020,2,25),date(2020,2,26),date(2020,4,10),date(2019,4,21),date(2020,5,1),date(2020,6,11),date(2020,8,15),date(2020,9,7),date(2020,10,12),date(2020,11,2),date(2020,11,15),date(2020,12,8),date(2020,12,25),date(2021,1,1),date(2021,2,15),date(2021,2,16),date(2021,2,17),date(2021,4,2),date(2021,4,21),date(2021,5,1),date(2021,6,3),date(2021,8,15),date(2021,9,7),date(2021,10,12),date(2021,11,2),date(2021,11,15),date(2021,12,8),date(2021,12,25)]
    db['Feriado']=db[' VIAGEM'].apply(lambda x: 1 if x in feriados else 0)
    dia=timedelta(1)
    pre_feriados=[]
    pos_feriados=[]
    for i in feriados:
        pre_feriados.append(i-dia)
        pos_feriados.append(i+dia)
    db['Pre Feriado']=db[' VIAGEM'].apply(lambda x: 1 if x in pre_feriados else 0)
    db['Pos Feriado']=db[' VIAGEM'].apply(lambda x: 1 if x in pos_feriados  else 0)
    
    db['Pandemia']=db[" VIAGEM"].map(lambda x: 1 if x>date(2020,3,15) else 0)

    #Substituindo os nomes das 319 linhas de ônibus por valores inteiro de 1 até 319
    db[' LINHA']=db[' LINHA'].astype(str)
    valores=pd.Series(list(range(1,320)))
    linhas_velhas=[]
    for i in db[' LINHA'].unique():
        linhas_velhas.append(i)
    db[' LINHA']=db[' LINHA'].replace(list(db[' LINHA'].unique()),list(range(1,320)))
    return db

# 4-> Compilar planilhas e agrupar os dados climáticos em um DataFrame(compilador_de_dados_climaticos)
def compilador_de_dados_climaticos (Clima,ano,diretorio_arquivos):
    #Nas planilhas de dados climáticos, antes de se rodar o código, excluiu-se as 8 primeiras linhas e todas colunas, com exceção de Data, Hora, Temperatura do Ar e Precipitação Total
    while ano<22:
        caminho=diretorio_arquivos+'/mg'+str(ano)+'.csv'
        df=pd.DataFrame(pd.read_csv(caminho,encoding='utf-8',sep=';'))
        Clima=pd.concat([df,Clima], ignore_index = True)
        ano+=1
    return Clima

# 5-> Formatação do DataFrame climático, substituindo valores anômalos de Chuva e Temperatura (formatador_clima)
def formatador_clima(db):
    db['Chuva']=db['Chuva'].str.replace(",",".")
    db['Temp']=db['Temp'].str.replace(",",".")
    
    #Substituindo valores '-9999' por valores nulos
    db['Chuva']=db['Chuva'].replace("-9999",np.nan)
    db['Temp']=db['Temp'].replace("-9999",np.nan)
    
    db['Chuva']=db['Chuva'].astype(float)
    db['Temp']=db['Temp'].astype(float)

    #Preenchendo valores nulos de chuva por 0 e valores nulos de temperatura pela média
    db['Chuva']=db['Chuva'].fillna(0)
    db['Temp']=db['Temp'].fillna(db['Temp'].mean())

    #Agrupando medições horárias por dia, e gerando novas colunas, do somatório total de chuva, além de valores médios, máximos e mínimos de temperatura
    db=db.groupby('Data', axis=0, as_index=False).agg({'Chuva':'sum','Temp':['mean','max','min']})
    db.columns=db.columns.droplevel(0)

    #Mexendo na coluna de data (VIAGEM) para criar compatibilidade com o DataFrame de passageiros, e juntar os dois DataFrames a partir dela
    db['']=pd.to_datetime(db[''],dayfirst=True)
    db=db.sort_values(by='')
    db=db.rename(columns={'':' VIAGEM'})

    return db

# 6-> Iterações que percorrem os dados, dividindo conjuntos de treino e teste, seguindo o modelo janela crescente com validação adiante (timesplit)
def timesplit(tscv,X,y,n_estimators,max_depth):
    #Modelo de Floresta Aleatória de Regressão
    reg= RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state = 42)
    resultados=[]
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
      start=time.time() #Início do cálculo do tempo decorrido

      #Índices de treino e teste de acordo com a iteração da janela crescente com validação adiante
      print("Fold: {}".format(fold))
      print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
      print("\n")

      #Divisão dos dados entre treino e teste, nos atributos (X) e variável 'Passageiros' (Y)
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      
      reg.fit(X_train, y_train) #Fit com valores de treino
      acc = round(reg.score(X_test,y_test)* 100, 2) #R² com valores de teste
      y_pred = reg.predict(X_test) #Prevendo valores de teste
      erro_ab=mean_absolute_error(y_test, y_pred) #MAE com valores preditos
      
      end=time.time() #Final do cálculo do tempo decorrido
      
      resultados.append([acc,erro_ab,train_index[-1],(end-start)])
      print("R2 do Regressor de Floresta Aleatória:",acc_rfr,"%")
      grafico_distribuição_previsoes(y_test,y_pred,X_train,max_depth,n_estimators) #Criação dos gráficos tipo scatter da distribuição das previsões frente aos valores reais

    return resultados

# 7-> Modelos de Floresta Aleatória com diferentes hiperparâmetros sendo iterados na função 6, e reunindo os resultados em um DataFrame (florestas_aleatorias)
def florestas_aleatorias(tscv, instancia, X, y):
    #Parâmetros de quantidade de árvores
    nos=[2,10,20,50,100]
    #Paâmetros de profundidade máxima da árvore
    profund=[5,10,20,50,100]
    Resultados=pd.DataFrame()
    for i in nos:
        for j in profund:
            #Salvando no DataFrame de resultados da função acima de acordo com os hiperparâmetros usados por cada modelo
            Resultados[instancia+' DATAFRAME, '+'A'+str(i)+' N'+str(j)]=timesplit(tscv,X,y,i,j)
    return Resultados

# 8-> Geração do gráfico que mostra os valores das previsões frente aos valores reais, e sua distribuição num gráfico tipo scatter (grafico_distribuição_previsões)
def grafico_distribuição_previsoes(y_test,y_pred,X_train,max_depth,n_estimators):
    df_fa = pd.DataFrame({'Valor Real': y_test, 'Previsão': y_pred})
    plt.scatter(y_test,y_pred) #Plot Scatter
    ax = sns.regplot(x="Valor Real", y="Previsão", data=df_fa, color='red',marker="") #Linha vermelha com regressão dos pontos, a linha verde do x=y foi traçada no GIMP

    #Salvando gráficos de acordo com a divisão de dados trabalhada
    if len(X)>410000:
        plt.savefig('TODO DATAFRAME A'+str(n_estimators)+" N"+str(max_depth)+' '+str(len(X_train))+'.png')
    elif len(X)<200000:
        plt.savefig('PAN DATAFRAME A'+str(n_estimators)+" N"+str(max_depth)+' '+str(len(X_train))+'.png')
    else:
        plt.savefig('PRE DATAFRAME A'+str(n_estimators)+" N"+str(max_depth)+' '+str(len(X_train))+'.png')
    plt.clf()

# 9-> Geração dos gráficos das métricas de desempenho do modelo A100 N100 no decorrer das iterações da função 6, além de chamar a função seguinte para mais gráficos (graficos_resultados)
def graficos_resultados(Resultados):
    #Dividindo as métricas a partir do DataFrame dos resultados
    Resultados_R2=Resultados.applymap(lambda x: x[0])
    Resultados_mae=Resultados.applymap(lambda x: x[1])
    Resultados_tempo=Resultados.applymap(lambda x: x[3])

    #Plotando com eixo secundário os valores de MAE e R² de acordo com iterações da janela para o modelo A100N100
    fig,ax = plt.subplots()
    ax.plot(Resultados_R2.iloc[:,-1],color="red", marker="o")
    ax.set_xlabel("Iterações da Janela",fontsize=14)
    ax.set_ylabel("R²",color="red",fontsize=14)
    ax2=ax.twinx()
    ax2.plot(Resultados_mae.iloc[:,-1],color="blue",marker="o")
    ax2.set_ylabel("Erro Médio Absoluto (Passageiros)",color="blue",fontsize=14)
    plt.tight_layout()
    plt.show()

    #Chamando a próxima função para plotar métricas de desempenho para todos modelos de Floresta Aleatória
    plotador_modelos(Resultados_R2,'box_r2')
    plotador_modelos(Resultados_mae,'box_mae')
    plotador_modelos(Resultados_tempo,'plt')    

# 10-> Geração dos gráficos dos desempenhos de MAE e R² de cada modelo de Floresta Aleatória em um boxplot, além do gráfico de tempo decorrido pelos modelos (plotador_modelos)
def plotador_modelos(db,tipo):
    db.columns=['A2 N5','A2 N10','A2 N20','A2 N50','A2 N100','A10 N5','A10 N10','A10 N20','A10 N50','A10 N100','A20 N5','A20 N10','A20 N20','A20 N50','A20 N100','A50 N5','A50 N10','A50 N20','A50 N50','A50 N100','A100 N5','A100 N10','A100 N20','A100 N50','A100 N100']
    sns.set_style("whitegrid")
    if tipo=='plt':
        #Plot do tempo decorrido por cada modelo de Floresta Aleatória
        db=db.sum()
        db.plot(fontsize=14)
        plt.xlabel('Modelos de Floresta Aleatória')
        plt.ylabel('Segundos')
        plt.tight_layout()
        plt.show()
        plt.clf()   
    else:
        #Boxplot dos valores de R² e MAE para cada modelo de Floresta Aleatória
        plt.boxplot(db)
        plt.xticks(ticks=range(len(db.columns)),labels=db.columns,rotation=45)  
        plt.xlabel('Modelos de Floresta Aleatória')
        if tipo=='box_r2':
            plt.ylabel('R²')
        else:
            plt.ylabel('Erro Médio Absoluto (Passageiros)')
        plt.tight_layout()
        plt.show()
        plt.clf()

#Chamando função 1 de compilação dos dados de passageiros
Passageiros=pd.DataFrame()
Passageiros=compilador_de_dados_passageiros(Passageiros,16,1,'C:/Users/unkno/Downloads/mco')

#Retirando valores anômalos com mais de 500 passageiros por viagem, além de valores zerados e negativos de passageiros
Passageiros[' VIAGEM']=pd.to_datetime(Passageiros[' VIAGEM'],dayfirst=True)
Passageiros=Passageiros.sort_values(by=' VIAGEM')

maiores=Passageiros[Passageiros.Passageiros>500]
Passageiros=Passageiros[Passageiros.Passageiros<500]
zeros=Passageiros[Passageiros.Passageiros==0]
Passageiros=Passageiros[Passageiros.Passageiros!=0]
negativos=Passageiros[Passageiros.Passageiros<0]
Passageiros=Passageiros[Passageiros.Passageiros>0]

#Gerando gráfico com a função 2 de todas as viagens, antes e depois destas serem agrupadas por dia
grafico_passageiros(Passageiros)
Passageiros=Passageiros.groupby([' VIAGEM',' LINHA'], axis=0, as_index=False).sum()
grafico_passageiros(Passageiros)

#Chamando função 3 para formatar o DataFrame de passageiros
Passageiros=formatador_passageiros(Passageiros)

#Chamando função 4 de compilação dos dados climáticos e função 5 para formatar
clima_df= pd.DataFrame()
clima_df=compilador_de_dados_climaticos(clima_df,16,'C:/Users/unkno/Desktop/resultados')
clima_df=formatador_clima(clima_df)

#Juntado os dados de passageiros com dados climáticos
df_final=pd.merge(Passageiros,clima_df,on=' VIAGEM')
df_final=df_final.rename(columns={'sum':'Chuva','mean':'Temperatura media','max':'Temperatura Max','min':'Temperatura Min'})
df_final=df_final.drop(" VIAGEM",axis=1)

#Separando todos atributos de clima e calendário(X), da variável do número de passageiros (Y), que será predita no método de Florestas Aleatórias
X=df_final.copy()
y=X.pop("Passageiros")

#Primeiro experimento sendo realizados no período completo de dados
tscv=TimeSeriesSplit(n_splits=24) #+- 3 meses
Resultados_periodocompleto=pd.DataFrame()
Resultados_periodocompleto=florestas_aleatorias(tscv,'TODO',X,y)

#Divindo o conjunto de dados para separar o período sem os anos pandêmicos (2016-2019) e o período com os anos pandêmicos (2020-2021)
X_pan=X[400038:]
y_pan=y[400038:]

X_pre=X[:400038]
y_pre=y[:400038]

#Segundo experimento sendo realizados no período de dados sem os anos pandêmicos
tscv=TimeSeriesSplit(n_splits=16) #+- 3 meses
Resultados_pre_pandemia=pd.DataFrame()
Resultados_pre_pandemia=florestas_aleatorias(tscv,'PRE',X_pre,y_pre)

#Terceiro experimento sendo realizados no período de dados com os anos pandêmicos
tscv=TimeSeriesSplit(n_splits=8) #+- 3 meses
Resultados_pan=pd.DataFrame()
Resultados_pan=florestas_aleatorias(tscv,'PAN',X_pan,y_pan)

#Gerando gráficos finais das métricas de desempenho para cada conjunto de experimento feito
graficos_resultados(Resultados_periodocompleto)
graficos_resultados(Resultados_pre_pandemia)
graficos_resultados(Resultados_pan)
