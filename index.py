from helpers.help import openCsv, normalizeData
from network.network import LSTM
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# IMPORTANDO E NORMALIZANDO OS DADOS DE TREINAMENTO
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

data = openCsv('data/DailyDelhiClimateTrain.csv')
x = data.loc[ : , data.columns != 'meantemp']
y = data.iloc[:, 0]

mmx = MinMaxScaler().fit(x)
mmy = MinMaxScaler().fit(y.values.reshape(-1, 1))

x_mm, y_mm = normalizeData('data/DailyDelhiClimateTrain.csv', [mmx, mmy])

# IMOPORTANDO E NORMALIZANDO OS DADOS DE VALIDAÇÃO
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

data = openCsv('data/DailyDelhiClimateTest.csv')

x_val = data.loc[ : , data.columns != 'meantemp']
y_val = data.iloc[:, 0]

mmx_val = MinMaxScaler().fit(x_val)
mmy_val = MinMaxScaler().fit(y_val.values.reshape(-1, 1))

x_mm_val, y_mm_val = normalizeData('data/DailyDelhiClimateTest.csv', [mmx_val, mmy_val])

data = {
    'train':{
        'x': x_mm,
        'y': y_mm
    },
    'validation':{
        'x': x_mm_val,
        'y': y_mm_val
    }
}

#carrega o modelo e suas variáveis globais
inputSize = 3
hiddenSize = 128
numLayers = 1
numOut = 1

#instancia a rede neural
net = LSTM(inputSize, hiddenSize, numLayers, numOut)

#instancia o otimizador e a função de erro
gradienteErro = torch.nn.MSELoss()

#informações para treinar a rede neural
epochs = 280
dbg = 0
error = 0
lr = 0.003

#treina o modelo
for i in range(epochs):
    
    epochError = []

    """ if i % 10 == 0 and i != 0:
        lr -= 0.01
    else:
        lr *= 0.80

    if lr < 0.005:
        lr = 0.003 """
    if i > 200:
        correcaoErro = torch.optim.Adamax(net.parameters(), lr=lr)
    else:
        lr *= 0.97
        correcaoErro = torch.optim.Adagrad(net.parameters(), lr=lr)


    """ if i < 200 and i != 0:
        correcaoErro = torch.optim.Adamax(net.parameters(), lr=lr)
        lr -= 0.0045
    else:
        lr *= 0.95
        if lr < 0.005:
            lr = 0.005
        correcaoErro = torch.optim.Adamax(net.parameters(), lr=lr) """

    for j in range(len(data['train']['x'])):
        
        #zera o gradiente de erro
        correcaoErro.zero_grad()

        #pega um conjunto de dados
        x = torch.tensor(data['train']['x'][j]).float()
        y = torch.tensor(data['train']['y'][j]).float()

        #redimensiona o conjunto de dados
        y = y.view(1,  -1)

        #print(f'{x} / {y}')

        #passa o conjunto de dados pela rede neural
        saida = net(x.view(1, 1, -1))

        #calcula o erro
        erro = gradienteErro(saida, y)

        #propaga o erro
        erro.backward()

        #atualiza os pesos
        correcaoErro.step()
        epochError.append(erro.item())
        if(dbg):
            if j % 10 == 0 and j == 0:
                print(f'Erro: {erro}')
                print(f'Saída: {saida}')
                print(f'Y: {y}')
                print(f'X: {x}')
                print(f'Época: {i}')
                print(f'Iteração interna: {j}')
                print('-----------------------------')

    print(f'Erro da época {i} = {np.array(epochError).sum()}')
net.eval()

with torch.no_grad():

    resps = []
    #valida o modelo
    for i in range(len(data['validation']['x'])):
        x = torch.tensor(data['validation']['x'][i]).float()
        y = torch.tensor(data['validation']['y'][i]).float()

        y = y.view(1, -1)

        saida = net(x.view(1, 1, -1))

        erro = gradienteErro(saida, y)

        resps.append(saida)

        print(f'Erro: {erro}')
        print(f'Saída: {saida}')
        print(f'Y: {y}')
        print(f'X: {x}')
        print(f'Iteração: {i}')
        print('-----------------------------')

    #mmRes = MinMaxScaler().fit(np.array(resps).reshape(-1, 1))
    resps = [mmy_val.inverse_transform(i) for i in resps]
    y_val = mmy_val.inverse_transform(data['validation']['y'])

    for item in range(0, len(y_val)):
        y_val[item] = y_val[item][0]

    for item in range(0, len(resps)):
        resps[item] = resps[item][0][0]
    
    plt.plot(resps, label='Resposta')
    plt.plot(y_val, label='Real')
    plt.legend()
    plt.savefig("graphic.png")
