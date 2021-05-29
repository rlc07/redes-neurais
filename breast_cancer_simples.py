
import pandas as pd
import keras
from keras.models import Sequential #classe responsavel por criar a rede neural com varias camadas ocultas
from keras.layers import Dense #classe responsavel por adicionar camadas densas a rede neural (cada um dos neuronios é ligado com todos os neuronios da camada subsequente)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv') # classsificação binaria 1 = tem tumor, 0 não tem

print(len(previsores))
print(len(classe))

#divisao da base de dados treinamento e teste
#treinamento = responsavel por identificar os pesos que deve ser usada
#teste = avaliação do percentual de acerto e erros
#test_size=0.25 representa que vamos usar 25% dos registros do csv para fazer os testes e 75% para treinar
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

#cria uma nova rede neural
classificador = Sequential()

#adiciona primeira camada oculta
#camada essa que sera ligada com todas as camadas subsequentes

#units = quantos neuronios farão parte da camada oculta
# - formula para descobrir quantos neuronios adicionar = numero_de_entradas(colunas do csv) + numeros_de_camadas / 2 (30 + 1(pq a saida é binaria)) / 2
#Temos que ir aumentando ou diminuindo essa quantidade conforme a musica anda

#activation = a função de ativação relu

#kernel_initializer = como sera feita a inicialização dos pesos padrão random_uniform

#input_dim = adiciona quantos elementos na camada de entrada (30 colunas csv) - somente na primeira camada oculta

classificador.add(Dense(units = 16, activation= 'relu',
                        kernel_initializer= 'random_uniform', input_dim = 30))
classificador.add(Dense(units = 16, activation= 'relu',
                        kernel_initializer= 'random_uniform'))
#adiciona camada de saida

#units = 1 como a resposta é (maligno ou benigno) se retornar valor proximo a 1 seignifica que o cancer é maligno caso contrario proximo a 0 benigno

#activation = como estamos trabalhando com classificador binario 0 ou 1 a função deve ser sigmoid
#usa a sigmoid que retorna a probabilidade de ser 0 ou 1

classificador.add(Dense(units = 1, activation = 'sigmoid'))

# rede criada 1 camada = https://github.com/rlc07/redes-neurais/blob/master/images/img.png
# rede criada 2 camada = https://github.com/rlc07/redes-neurais/blob/master/images/img_1.png

#compila rede neural

# optimizer = função usada para fazer os ajustes do peso se os resultados não for bom podemos testar com outros
# loss função de erro  para classificação binaria 0 ou 1 usar binary_crossentropy para classificar ims exemplo gato, cachorro ou rato podemos usar categorical_crossentropy
# metrics = metricas para fazer avaliação
#  - binary_accucary = pega quantidade de registros que deram sucesso e erros e faz calculo da diferença

#classificador.compile(optimizer = 'adam', loss='binary_crossentropy', metrics= ['binary_accuracy'])

otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#executa o treinamento com a base de dados de treinamento
# batch_size = atualiza os pesos a cada 10 registros
# epochs = quantas vezes é executada os ajustes dos pesos
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)

# mostra todos os pesos encontrato no treinamento
peso0 = classificador.layers[0].get_weights()
peso1 = classificador.layers[1].get_weights()
peso2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(previsores_teste)

previsoes = (previsoes > 0.5)

#caucula a porcentagem de acerto do algoritomo em cima da base de teste retornando 89% de acertos sobre a base de teste
precisao = accuracy_score(classe_teste, previsoes)
print(precisao)

#faz uma avaliação de quais classe o algoritimo esta certando e errando
#exemplo 0 = 100 1 = 30 erros etc..
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)
#ultiliza o keras msm responsabilidade acima
resultado = classificador.evaluate(previsores_teste, classe_teste)

