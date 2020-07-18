######################################################################################################################
# - UNIVERSIDAD: UNIVERSIDAD DE GRANADA.
#
# - FACULTAD: ESCUELA TÉCNICA SUPERIOR DE INGENIERÍAS INFORMÁTICA Y TELECOMUNICACIÓN.
#
# - CURSO: MASTER PROFESIONAL EN INGENERÍA INFORMÁTICA. 2019-2020.
#
# - ASIGNATURA: TRABAJO FIN DE MÁSTER: DEEP LEARNING PARA SIMULACIÖN ENERGÉTICA DE EDIFICIOS.
#
# - NOMBRE: JESÚS MESA GONZÁLEZ.
#
# - FECHA: 07/07/2020.
#
# - XGBoost.py: Script que contiene la clase XGBoostTimeSeries, la cual proporciona una interfaz de alto nivel con
#   los atributos y métodos necesarios para:
#
#
#              1.- Definir un problema de predicción multistep de serie temporal multivariable, estableciendo para ello
#               el conjunto de datos que almacena la serie temporal, la variable objetivo que se desea predecir, el
#               valor de los parámetros lookback y delay y las fechas de inicio y fin de entrenamiento y validación.
#
#              2.- Definir un modelo de XGBoost mediante el uso de una estrategia directa para afrontar el problema
#              definido estableciendo su número de árboles, su profundidad máxima y la tasa de aprendizaje.
#
#              3.- Entrenar y validar el modelo de XGBoost definido. Generando para ello las muestras de entrenamiento y
#              validación.
#
#              4.- Evaluar visualmente el modelo de XGBoost entrenado pintando en una gráfica los valores predichos de
#              la variable objetivo a partir de las muestras de validación junto a los valores verdaderos.
######################################################################################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class XGBoostTimeSeries:

    # Constructor de la clase que declara e inicializa los atributos, los cuales son:
    #
    #   - dataframe: Pandas dataframe con la serie temporal que se utilizará para entrenar y validar el modelo.
    #   - target: String con el nombre de la columna del conjunto de datos (dataframe) que contiene los valores
    #   de la variable objetivo.
    #   - lookback: Número de timesteps que retrocedemos hacia atrás a partir de un timestep concreto de la serie
    #   temporal a la hora de crear una secuencia para entrenar y validar el modelo.
    #   - delay: Número de timesteps después del último timestep de la secuencia de entrada para los que queremos
    #   que la red neuronal prediga la variable objetivo.
    #   - step: Índica el muestreo de los timesteps a la hora de crear una secuencia para entrenar y validar
    #   el modelo, de forma que las secuencias de entrada de la red tendrán longitud lookback // step, mientras que
    #   las secuencias objetivo delay // step.
    #   - n_estimators: Número de árboles que tendrá cada modelo de xgb aprendido para predecir el valor de las
    #   variables objetivo en un timestep fututo concreto.
    #   - max_depth: Máxima profundidad de cada árbol de cada modelo.
    #   - learning_rate: Tasa de aprendizaje durante el entrenamiento de cada árbol.
    def __init__(self, dataframe, target, lookback, delay, date_train_ini, date_train_end, date_val_ini, date_val_end,
                  n_estimators, max_depth, learning_rate):

        # Inicializamos los atributos.
        self._dataframe = dataframe
        self._target = target
        self._lookback = lookback
        self._delay = delay
        self._date_train_ini = date_train_ini
        self._date_train_end = date_train_end
        self._date_val_ini = date_val_ini
        self._date_val_end = date_val_end
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate

        # Normalizamos los datos.
        scaler = StandardScaler()
        scaler.fit(self._dataframe.loc[self._date_train_ini:self._date_train_end].values)
        norm_data = scaler.transform(self._dataframe.loc[self._date_train_ini:self._date_val_end].values)
        self._norm_data = pd.DataFrame(data=norm_data,
                                       index=self._dataframe.loc[self._date_train_ini:self._date_val_end].index,
                                       columns=self._dataframe.loc[self._date_train_ini:self._date_val_end].columns)

        # Definimos el modelo.
        self.defineXGBoostModel()

    # Método para consultar el conjunto de datos.
    def getDataFrame(self):
        return (self._dataframe)

    # Método para establecer el target.
    def setTarget(self, target):
        self._target = target

    # Método para consultar el target.
    def getTarget(self):
        return (self._target)

    # Método para establecer el lookback.
    def setLookBack(self, lookback):
        self._lookback = lookback

    # Método para consultar el lookback.
    def getLookBack(self):
        return (self._lookback)

    # Método para establecer el delay.
    def setDelay(self, delay):
        self._delay = delay

    # Método para consultar el delay.
    def getDelay(self):
        return (self._delay)

    # Método para consultar la fecha de inicio de entrenamiento.
    def getDateTrainIni(self):
        return (self._date_train_ini)

    # Método para consultar la fecha de fin de entrenamiento.
    def getDateTrainEnd(self):
        return (self._date_train_end)

    # Método para consultar la fecha de inicio de validación.
    def getDateValIni(self):
        return (self._date_val_ini)

    # Método para consultar la fecha de fin de validación.
    def getDateValEnd(self):
        return (self._date_val_end)

    # Método para establecer el número de estimadores del modelo.
    def setNEstimators(self, n_estimators):
        self._n_estimators = n_estimators

        # Es necesario redefinir el modelo.
        self.defineXGBoostModel()

    # Método para consultar el número de estimadores del modelo.
    def getNEstimators(self):
        return (self._n_estimators)

    # Método para establecer la profundidad de los estimadores del modelo.
    def setMaxDepth(self, max_depth):
        self._max_depth = max_depth

        # Es necesario redefinir el modelo.
        self.defineXGBoostModel()

    # Método para consultar la profundidad de los estimadores del modelo.
    def getMaxDepth(self):
        return (self._max_depth)

    # Método para establecer la tasa de aprendizaje del modelo.
    def setLearningRate(self, learning_rate):
        self._learning_rate = learning_rate

        # Es necesario redefinir el modelo.
        self.defineXGBoostModel()

    # Método para consultar la tasa de aprendizaje del modelo.
    def getLearngingRate(self):
        return (self._learning_rate)

    # Método para consultar el modelo.
    def getModel(self):

        return(self._model)

    # Método para consultar el conjunto de datos normalizado.
    def getNormData(self):

        return(self._norm_data)

    # Método que define el modelo de XGBoost con el que vamos a abordar el problema de predicción.
    def defineXGBoostModel(self):
        self._model = MultiOutputRegressor(xgb.XGBRegressor(objective="reg:squarederror",
                                                            n_estimators=self._n_estimators,
                                                            max_depth=self._max_depth,
                                                            learning_rate=self._learning_rate),
                                           n_jobs=12)

    # Método para generar un conjunto de muestras de entrenamiento junto con sus targets correspondientes a partir
    # de los timesteps del conjunto de datos de entrenamiento asociado al objeto implícito de esta clase situados
    # entre una fecha inicial y otra final.
    #
    #   - Parámetros:
    #
    #       - date_ini: Fecha inicial.
    #       - date_end: Fecha final.
    #
    #   - Valor devuelto.
    #
    #       - train_x: Matriz que en cada fila contiene una muestra en forma de serie temporal multivariable aplanada.
    #       - train_y: Matriz que en la fila i contiene los delay valores de la variable objetivo que vienen después
    #       del último timestep de la muestra i de train_x.
    def generateTrainSamples(self, date_ini, date_end):

        # En primer lugar nos quedamos con los timesteps entre date_ini y date_end.
        data = self._norm_data.loc[date_ini:date_end]

        # Recorremos los timesteps con los que podemos generar muestras.
        indexes = np.arange(self._lookback, len(data) - self._delay)

        X = np.zeros(shape=(len(indexes), (self._lookback + 1) * self._norm_data.values.shape[-1]))
        Y = np.zeros(shape=(len(indexes), self._delay))

        for i, idx in enumerate(indexes):

            # Generamos la muestra (serie temporal) a partir del timestep con índice idx.
            seq = data.iloc[np.arange(idx - self._lookback, idx + 1)].values

            # Aplanamos la muestra y la ponemos en la fila i de la matriz en la que almacenamos los datos de
            # entrenamiento
            X[i] = seq.flatten()

            # Obtenemos el vector de valores de la variable objetivo que le corresponden a esta muestra i-ésima.
            Y[i] = data[self._target].iloc[np.arange(idx + 1, idx + 1 + self._delay)].values.flatten()

        return (X, Y)

    # Método para generar un conjunto de muestras de validación junto con sus targets correspondientes a partir
    # de los timesteps del conjunto de datos de validación asociado al objeto implícito de esta clase situados
    # entre una fecha inicial y otra final.
    #
    #   - Parámetros:
    #
    #       - date_ini: Fecha inicial.
    #       - date_end: Fecha final.
    #
    #   - Valor devuelto.
    #
    #       - train_x: Matriz que en cada fila contiene una muestra en forma de serie temporal multivariable aplanada.
    #       - train_y: Matriz que en la fila i contiene los delay valores de la variable objetivo que vienen después
    #       del último timestep de la muestra i de train_x.
    def generateValSamples(self, date_ini, date_end):

        # En primer lugar nos quedamos con los timesteps entre date_ini y date_end.
        data = self._norm_data.iloc[
               (self._norm_data.index.get_loc(date_ini) - self._lookback):self._norm_data.index.get_loc(date_end)]

        # Recorremos los timesteps con los que podemos generar muestras.
        indexes = np.arange(self._lookback, len(data) - self._delay)

        X = np.zeros(shape=(len(indexes), (self._lookback + 1) * self._norm_data.values.shape[-1]))
        Y = np.zeros(shape=(len(indexes), self._delay))

        for i, idx in enumerate(indexes):
            # Generamos la muestra (serie temporal) a partir del timestep con índice idx.
            seq = data.iloc[np.arange(idx - self._lookback, idx + 1)].values

            # Aplanamos la muestra y la ponemos en la fila i de la matriz en la que almacenamos los datos de
            # entrenamiento
            X[i] = seq.flatten()

            # Obtenemos el vector de valores de la variable objetivo que le corresponden a esta muestra i-ésima.
            Y[i] = data[self._target].iloc[np.arange(idx + 1, idx + 1 + self._delay)].values.flatten()

        return (X, Y)

    # Método para entrenar el modelo de XGBoost.
    def train(self):
        # Generamos las muestras de entrenamiento.
        train_x, train_y = self.generateTrainSamples(self._date_train_ini, self._date_train_end)

        # Finalmente entrenamos el modelo.
        self._model.fit(train_x, train_y)

    # Método para evaluar en términos de NMAE el rendimiento del modelo sobre series temporales de validación.
    def evaluateModel(self):

        # Generamos las muestras (series temporales de validación).
        val_x, val_y = self.generateValSamples(self._date_val_ini, self._date_val_end)

        # Predecimos las muestras de validación con el modelo.
        preds_y = self._model.predict(val_x)

        return (np.mean(np.abs(val_y - preds_y)))

    # Función que genera una serie temporal de longitud lookback a partir de un timestep concreto recibido como
    # argumento y a partir de ella predice los siguientes delay valores de la variable objetivo (target) con el modelo
    # actual, pintando estos últimos en una gráfica junto con los valores verdaderos.
    #
    #   - Parámetros:
    #
    #       - timestep: String con el instante de tiempo del conjunto de datos dataframe asociado al objeto de esta
    #       clase a partir del cual vamos a crear la serie temporal a predecir. Formato: yyyy-mm-dd HH:MM:SS.
    #       - title: Título de la gráfica.
    def plot_prediction_half_day(self, timestep, title):
        # Índice del timestep.
        i = self._norm_data.index.get_loc(timestep)

        # Generamos la muestra (serie temporal) a partir del timestep con índice i.
        seq = self._norm_data.iloc[np.arange(i - self._lookback, i + 1)].values
        seq = seq.flatten()

        # Generamos el vector de valores deseados de la variable objetivo.
        targets = self._norm_data[self._target].iloc[np.arange(i + 1, i + 1 + self._delay)].values.flatten()

        # Predecimos la serie temporal generada.
        prediction = self._model.predict(seq.reshape((1, len(seq))))[0]

        # Pintamos los valores verdaderos y los predichos por el modelo.
        plt.figure()

        plt.plot(self._norm_data.index[np.arange(i + 1, i + 1 + self._delay)], prediction)
        plt.plot(self._norm_data.index[np.arange(i + 1, i + 1 + self._delay)], targets)

        plt.title(title)
        plt.xlabel("Tiempo")
        plt.ylabel(self._target + " Consumo")
        plt.legend(["Predicción", "Verdadero"],bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

        plt.show()

    # Método que predice con el modelo actual todos los valores de validación de la variable objetivo y los pinta
    # en una gráfica junto a los valores verdaderos.
    #
    #   - Parámetros:
    #
    #       - title: Título de la gráfica.
    def plot_prediction(self, title):

        # Generamos los índices de los timesteps del conjunto de datos que pertenecen al conjunto de validación y
        # que son susceptibles de crear muestras de validación.
        data = self._norm_data.iloc[(self._norm_data.index.get_loc(self._date_val_ini) - self._lookback):self._norm_data.index.get_loc(self._date_val_end)]
        indexes = np.arange(self._lookback, len(data) - self._delay, self._delay)

        # Listas en las que almacenar los valores predichos por el modelo y los verdaderos. Además de las fechas de
        # dichos datos.
        predicted_values = []
        true_values = []
        date_values = []

        for i in indexes:

            # Creamos la secuencia.
            seq = data.iloc[np.arange(i - self._lookback, i + 1)].values
            seq = seq.flatten()

            targ = data.iloc[np.arange(i + 1, i + self._delay + 1)][self._target].values
            dates = data.iloc[np.arange(i + 1, i + self._delay + 1)].index.strftime("%Y-%m-%d %H:%M:%S").values

            # Predecimos los siguientes valores a partir de la muestra generada.
            pred = self._model.predict(seq.reshape((1, len(seq))))[0]

            # Añadimos los valores predichos y los verdaderos a su corresondientes listas.
            predicted_values += pred.tolist()
            true_values += targ.tolist()
            date_values += dates.tolist()

        # Pintamos los valores verdaderos y los predichos por el modelo.
        plt.figure(dpi=500)
        plt.style.use("seaborn-darkgrid")

        plt.plot(date_values, predicted_values)
        plt.plot(date_values, true_values)

        plt.title(title)
        plt.xlabel("Tiempo")
        plt.xticks(np.arange(0, len(date_values), 288))
        plt.ylabel(self._target + " Consumo")
        plt.legend(["Predicción", "Verdadero"], bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

        plt.show()