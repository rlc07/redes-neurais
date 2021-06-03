import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential #classe responsavel por criar a rede neural com varias camadas ocultas
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('ires.csv')

previsores = base.iloc[:, 0:4].values #pega os atributos
classe = base.iloc[:, 4].values

labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 '
classe_dumy = np_utils.to_categorical(classe)

def criar_rede():
    classificador = Sequential()

    classificador.add(Dense(units=4, activation='relu', input_dim=4))
    classificador.add(Dense(units=4, activation='relu'))

    # cria neuronios de saida
    classificador.add(Dense(units=3, activation='softmax'))

    classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn=criar_rede,
                                epochs = 1000,
                                batch_size = 10)
resultados = cross_val_score(estimator=classificador,
                             X=previsores, y=classe,
                             cv=10, scoring= 'accuracy')

media = resultados.mean()
desvio = resultados.std()