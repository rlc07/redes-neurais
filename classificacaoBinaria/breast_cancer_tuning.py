
import pandas as pd
import keras
from keras.models import Sequential #classe responsavel por criar a rede neural com varias camadas ocultas
from keras.layers import Dense, Dropout #classe responsavel por adicionar camadas densas a rede neural (cada um dos neuronios é ligado com todos os neuronios da camada subsequente)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv') # classsificação binaria 1 = tem tumor, 0 não tem

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation= activation,
                        kernel_initializer= kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2)) #zera 20% valores dos neuronios da camada

    classificador.add(Dense(units = neurons, activation= activation,
                        kernel_initializer= kernel_initializer))

    classificador.add(Dropout(0.2)) #zera 20% valores dos neuronios da camada

    classificador.add(Dense(units=1, activation='sigmoid'))
    classificador.compile(optimizer=optimizer, loss = loos, metrics=['binary_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn = criarRede)

#X = atributos previsores
# cv = quantas vezes sera executado o teste quebrando a base
parametros = { 'batch_size': [10, 30],
               'epochs': [50, 100],
               'optimizer': ['adam', 'sgd'],
               'loos': ['binary_crossentropy', 'hinge'],
               'kernel_initializer': ['random_uniform', 'normal'],
               'activation': ['relu', 'tanh'],
               'neurons': [16, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid=parametros,
                           scoring='accuracy',
                           cv=5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_




