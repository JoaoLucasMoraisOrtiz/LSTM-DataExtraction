import torch
import torch.nn as nn
import torch.nn.functional as functional

#nossa rede neural recorrente
class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_out):
        """ 
            função que inicializa a rede neural recorrente
            @param input_size: tamanho da entrada
            @param hidden_size: tamanho da camada escondida
            @param num_layers: número de camadas
            @param num_out: número de classes
        """

        #inicializa a nossa rede neural como sendo uma RNN
        super(LSTM, self).__init__()

        #tamanho do vetor da nossa memória de curto prazo
        self.hidden_size = hidden_size

        #número de camadas aninhadas (não número de iterações)
        self.num_layers = 1

        #nossa camada lstm
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        #camadas fully connecteds, utilizada para complementar a saída da lstm.
        self.fc = nn.Linear(hidden_size, 728)
        self.fc2 = nn.Linear(728, num_out)
        #self.fc3 = nn.Linear(102, 1)

    def forward(self, x):
        """
            função que realiza a passagem para frente da rede neural
            @param x: entrada da rede neural
            @return: saída da rede neural
        """

        #inicializa a memória da lstm para dados em batch
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        #inicializa a memória da lstm para dados individuais
        #h0 = torch.zeros(self.num_layers, self.hidden_size)
        #c0 = torch.zeros(self.num_layers, self.hidden_size)
        #passa a entrada pela lstm
        #outputs é a última saída da rede, hn são todas as saídas da rede.
        outputs, (hn, cn) = self.lstm(x, (h0, c0))
        #outputs = outputs.squeeze(0)
        #outputs, (hn, cn) = self.lstm2(outputs, (hn, cn))
        
        x = functional.leaky_relu(self.fc(outputs.squeeze(0)))
        #x, (hn, cn) = self.lstm2(outputs, (hn, cn))

        #passa a saída da lstm pela camada fully connected  
        x = functional.gelu(self.fc2(x))
        
        return x