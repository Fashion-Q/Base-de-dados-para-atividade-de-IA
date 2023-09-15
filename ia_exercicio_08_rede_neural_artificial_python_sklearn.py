import pandas as pd
# Carregando dados do arquivo CSV
url = 'https://raw.githubusercontent.com/Fashion-Q/Base-de-dados-para-atividade-de-IA/main/Base%20de%20dados.csv'
base_Treinamento = pd.read_csv(url,sep=';', encoding = 'latin1').values
print("---------------------------------")
print("Dados dos Pacientes - TREINAMENTO")
print("---------------------------------")
print(base_Treinamento)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede
print("Atributos de Entrada")
print("---------------------------------")
print(base_Treinamento[:, 1:8])

print("----------------------------")
print("Classificação Supervisionada")
print("----------------------------")
print(base_Treinamento[:, 8])

"""### Pré-processamento de Dados"""

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
# Binarizador de rótulo
lb = preprocessing.LabelBinarizer()

#Converter todos os atributos menos o outcome(ultima linha) em um array numPy
data = np.array(base_Treinamento)

sintomas = data[:, :-1]
diagnostico = data[:, -1]

print("--------------------------------")
print("Atributos de Entrada - Numéricos")
print("--------------------------------")
print(sintomas)

print("----------------------------------------")
print("Classificação Supervisionada - Numéricos")
print("----------------------------------------")
print(diagnostico)

"""### Treinamento do Neurônio Perceptron"""

from sklearn.linear_model import Perceptron
# Treinamento do Perceptron a partir dos atributos de entrada e classificações
modelo = Perceptron()
modelo.fit(sintomas, diagnostico)

# Acurácia do modelo, que é : 1 - (predições erradas / total de predições)
# Acurácia do modelo: indica uma performance geral do modelo.
# Dentre todas as classificações, quantas o modelo classificou corretamente;
# (VP+VN)/N
print('Acurácia: %.3f' % modelo.score(sintomas, diagnostico))

"""### ----------------------------------------------------------------------------

# Validação do Aprendizado

### Predição Simples
"""

Luiza = [[50,180,90,20,420,35,0.666,69,]]
print("Luiza", modelo.predict(Luiza))

Galvona = [[0,80,58,15,200,23,0.25,25,]]
print("Galvona", modelo.predict(Galvona))

"""### Predição a partir de base de dados (.csv)"""

import pandas as pd
# Carregando dados do arquivo CSV
url = 'https://raw.githubusercontent.com/Fashion-Q/Base-de-dados-para-atividade-de-IA/main/Base%20de%20dados.csv'
base_Testes = pd.read_csv(url,sep=',', encoding = 'latin1').values
print("----------------------------")
print("Dados dos Pacientes - TESTES")
print("----------------------------")
print(base_Testes)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede
print("Atributos de Entrada")
print("---------------------------------")
print(base_Testes[:, 0:7])

"""### Pré-processamento de Dados"""

import numpy as np
from sklearn import preprocessing

# Binarizador de rótulo
lb = preprocessing.LabelBinarizer()

#Converter todos os atributos menos o outcome(ultima linha) em um array numPy
data = np.array(base_Testes)

sintomas = data[:, :-1]
diagnostico = data[:, -1]



print("--------------------------------")
print("Atributos de Entrada - Numéricos")
print("--------------------------------")
print(sintomas)



"""### Predição da Base"""

base_Predicao = modelo.predict((sintomas))
print("Classificações: ", base_Predicao)

"""### Retorno aos valores Categóricos"""

#O ultimo passo nao foi feito porque não foi dado em binário
#e não sabiamos resolver