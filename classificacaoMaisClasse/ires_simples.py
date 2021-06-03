import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential #classe responsavel por criar a rede neural com varias camadas ocultas
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix

base = pd.read_csv('ires.csv')

previsores = base.iloc[:, 0:4].values #pega os atributos
classe = base.iloc[:, 4].values

labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 '
classe_dumy = np_utils.to_categorical(classe)


#divide as bases treinamento e teste
# test_size=0;25 25% dabase para testar e o restante para treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dumy, test_size=0.25)

#cria rede neural
#images/img_2.png
classificador = Sequential()

classificador.add(Dense(units = 4, activation= 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation= 'relu'))

#cria neuronios de saida
classificador.add(Dense(units = 3, activation= 'softmax'))

classificador.compile(optimizer='adam', loss= 'categorical_crossentropy',
                      metrics= ['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)

previsoes = (previsoes > 0.5)
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)