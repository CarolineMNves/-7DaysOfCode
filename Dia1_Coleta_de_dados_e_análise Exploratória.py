# %% [markdown]
# 1- importando as bibliotecas

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches

#aperfeiçoando as visualizações

%matplotlib inline
mpl.style.use('ggplot')
plt.style.use('fivethirtyeight')
sns.set(context= 'notebook', palette= 'dark', color_codes= True)

# %% [markdown]
# 2- lendo o arquivo
# 
# fonte Kaggle: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?utm_source=ActiveCampaign&utm_medium=email&utm_content=%237DaysOfCode+-+Machine+Learning+1%2F7%3A+Coleta+de+dados+e+An%C3%A1lise+Explorat%C3%B3ria&utm_campaign=%5BAlura+%237Days+Of+Code%5D%28Js+e+DOM+-+3%C2%AA+Ed+%29+1%2F7

# %%
spotify_data = pd.read_csv('C:/Users/Caroline/Downloads/archive (5)/dataset_spotify.csv')
spotify_data.head()

# %% [markdown]
# 3- verificando os dados
# 

# %%
spotify_data.info()

# %% [markdown]
# 4- descrição dos atributos numéricos

# %%
spotify_data.describe()

# %% [markdown]
# 5- Limpeza inicial 

# %%
#removendo a coluna Unnamed: 0
spotify_data = spotify_data.drop(columns= ['Unnamed: 0'])
spotify_data.head()

# %% [markdown]
# 6- Descrição dos dados 
# 

# %%
#observando a disposição dos dados e possíveis valores ausentes
spotify_data.shape

# %%
#quantidade de dados em cada coluna e seus respectivos formatos

spotify_data.info()

# %% [markdown]
# 7- Análise estatística 

# %%
#visualizando as estatísticas dos dados 
spotify_data.describe()

# %%
#descrição dos artistas e contagem de músicas por artista
print(spotify_data['artists'].unique().shape)
print(spotify_data['artists'].value_counts())

# %%
#Descobrindo valores ausentes e ordenando, em ordem decrescente, as variantes por seus valores faltantes
(spotify_data.isnull().sum() / spotify_data.shape[0]).sort_values(ascending= False)

# %%
spotify_data.isnull().sum()

# %%
#plotando o gráfico para interpretação dos dados faltantes, ou seja, 
#observando a frequência de dados ausentes em cada coluna
def missing_data(df) :
    values_isnull = df.isnull().sum()
    columns = df.columns
    dic = {"colunas":[], "values_isnull":[]}
    for coluna, quant in zip(columns, values_isnull):
        if quant > 0:
            dic['colunas'].append(quant)
            dic['values_isnull'].append(coluna)
    df = pd.DataFrame(dic)
    plt.figure(figsize= (15,5))
    sns.barplot(x= df["values_isnull"], y= df["colunas"], data= df, palette = "tab10")
    plt.xticks(rotation=45);

# %%
missing_data(spotify_data)

# %%
#Visualizando o Top 100 de músicas mais populares, com relação ao conjunto de dados

classificacao_df = spotify_data.sort_values('popularity', ascending= False).head(100)
classificacao_df.head()

# %%
#checando valores ausentes

spotify_data.isna().sum()

# %%
spotify_data.shape


# %%
spotify_data.columns

# %%
#Descobrindo os artistas mais populares
most_popular_artists = spotify_data[['artists', 'popularity']]
most_popular_artists = most_popular_artists.groupby('artists').mean().sort_values(by='popularity', ascending= False).reset_index()
most_popular_artists = most_popular_artists.head(10)
most_popular_artists

# %%
plt.barh(most_popular_artists['artists'], most_popular_artists['popularity'], color='green')
plt.title('The most popular artists')
plt.xlabel('Popularity')
plt.ylabel('Artists')
plt.tick_params(axis='y', labelsize=10)
plt.gca().invert_yaxis()
plt.tight_layout()

# Adicionando os rótulos nas barras
for i, v in enumerate(most_popular_artists['popularity']):
    plt.text(v + 0.5, i, str(v), color='black', fontweight='bold')

plt.show()


# %%
#top 10 musicas mais longas

long_musics = spotify_data[['track_name', 'duration_ms']].sort_values(by= "duration_ms", ascending= False)[:10]
long_musics

# %%
#plotando o gráfico
tp_lgst = sns.barplot(x= "duration_ms", y= "track_name", data= long_musics, color= "blue")
plt.title("Top 10 longest musics")
tp_lgst.set_xlabel("Duration (ms)")
tp_lgst.set_ylabel("Music")
plt.show

# %%
#top 10 de genêros mais populares

trend_genre= spotify_data[["track_genre", "popularity"]].sort_values(by= "popularity", ascending=False) [:10]
trend_genre

# %%
#plotando o gráfico
tp_genre = sns.barplot(x= "track_genre", y= "popularity", data= trend_genre, color="blue")
plt.title("Top trending genre")
tp_genre.set_xlabel("Genre")
tp_genre.set_ylabel("Popularity")
plt.show

# %%
#Top 10 musicas mais dançáveis 

most_danceable= spotify_data[["track_name", "artists", "danceability"]].sort_values(by="danceability", ascending=False) [:10]
most_danceable

# %%
#plotando o gráfico
plt.pie(x= "danceability", data= most_danceable, autopct='%1.2f%%', labels=most_danceable.track_name)
plt.title("Top 10 most danceable musics")
plt.show

# %%
#descobrindo se existe correlação entre as variáveis 
correlacao_spotify = spotify_data.corr(method= "pearson")
correlacao_spotify

# %%
#plotando a tabela de visualização da correlação
plt.figure(figsize=(15,4))
sns.heatmap(correlacao_spotify, annot=True, fmt=".1g")
plt.title("Correlation between variables")
plt.show()


