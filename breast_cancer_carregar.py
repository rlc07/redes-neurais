import numpy as np
from keras.models import model_from_json
import pandas as pd

arquivo = open('classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')


novo = np.array([[15.80, 8.34, 118, 100, 40, 600, 400, 322, 16.09, 10,
                  11.30, 11, 13, 14, 17, 19, 190, 470, 99, 12,
                  132.90, 66.9, 324.90, 34, 12, 13, 15, 16, 18, 19]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

print(previsao)
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

resultado = classificador.evaluate(previsores, classe)
print(resultado)
