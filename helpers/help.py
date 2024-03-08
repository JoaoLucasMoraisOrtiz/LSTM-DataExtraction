import pandas as pd
from pathlib import Path


def openCsv(dir):
    """ 
        Abre um arquivo csv e retorna uma lista com os dados do arquivo
        @param dir: string - Diretório do arquivo csv
        @return data: list - Lista com os dados do arquivo
    """
    root = str(Path.cwd())
    df = pd.read_csv(root + '/'+dir, index_col = 'date', parse_dates=True)
    return df

def normalizeData(dir, method, shuffle=True):
    """ 
        Normaliza os dados de um arquivo csv
        @param dir: string - Diretório do arquivo csv
    """

    #abre o arquivo csv
    df = openCsv(dir)

    if shuffle:
        df.sample(frac=1)

    #pega todos os dados de todas as linhas com excessão da última coluna de todas as linhas.
    x = df.loc[ : , df.columns != 'meantemp']

    #pega todos os dados de todas as linhas, mas apenas as colunas 5 e 6 de cada linha.
    y = df.iloc[:, 0]
    y = y.values.reshape(-1, 1)
    """ 
        vamos normalizar os dados para trabalharmos de forma a evidar problemas como gradiente explodindo ou desaparecendo.
        Já fizemos isto em outros projetos, como o daily-bike (https://github.com/JoaoLucasMoraisOrtiz/daily-bike-share-AI900/blob/main/helpers/help.py),
        mas não utilizamos a biblioteca sklearn naquelas ocasiões.
    """
    method[0].fit(x)
    method[1].fit(y)
    
    #valores de x e y normalizados
    x_ss = method[0].fit_transform(x)
    y_mm = method[1].fit_transform(y)

    return x_ss, y_mm
