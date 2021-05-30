
import pandas as pd
import keras
from keras.models import Sequential #classe responsavel por criar a rede neural com varias camadas ocultas
from keras.layers import Dense, Dropout #classe responsavel por adicionar camadas densas a rede neural (cada um dos neuronios é ligado com todos os neuronios da camada subsequente)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv') # classsificação binaria 1 = tem tumor, 0 não tem

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation= 'relu',
                        kernel_initializer= 'random_uniform', input_dim = 30))
    classificador.add(Dropout(0.2)) #zera 20% valores dos neuronios da camada

    classificador.add(Dense(units = 16, activation= 'relu',
                        kernel_initializer= 'random_uniform'))

    classificador.add(Dropout(0.2)) #zera 20% valores dos neuronios da camada

    classificador.add(Dense(units=1, activation='sigmoid'))
    otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    classificador.compile(otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)

#X = atributos previsores
# cv = quantas vezes sera executado o teste quebrando a base
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring= 'accuracy')

media = resultados.mean()
desvio = resultados.std()

print(media)




