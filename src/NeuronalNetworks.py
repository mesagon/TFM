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
# - NeuronalNetworks.py: Script que contiene tres clases empleadas para definir y experimentar con los modelos de
#   redes neuronales empleados en este trabajo. Estas clases son las siguientes:
#
#       - BasicTimeSeriesANN: Clase que proporciona una interfaz de alto nivel con los atributos y métodos necesarios
#       para:
#
#              1.- Definir un problema de predicción multistep de serie temporal multivariable, estableciendo para ello
#               el conjunto de datos que almacena la serie temporal, la variable objetivo que se desea predecir, el
#               valor de los parámetros lookback y delay y las fechas de inicio y fin de entrenamiento y validación.
#
#              2.- Definir una arquitectura de RNA secuencial (MLP, RNN o CNN) con
#               la que abordar el problema, estableciendo sus capas de entrada y de salida, las capas ocultas y
#               sus hiperparámtros y su función de error y algoritmo de optimización.
#
#              3.- Entrenar y validar el modelo de red neuronal definido. Para ello la clase establece
#              el tamaño de los lotes de entrenamiento y validación, el número de épocas y si se muestran a la red las
#              muestras de entrenamiento y validación en orden aleatorio o cronológico, tras lo cual genera con dichos
#              parámetros los conjuntos de muestras de entrenamiento y validación con los cuales finalmente se
#              entrena y valida respectivamente.
#
#              4.- Evaluar visualmente el modelo de RNA entrenado pintando en una gráfica los valores predichos de la
#              variable objetivo a partir de las muestras de validación junto a los valores verdaderos.
#
#       - BasicTimeSeriesANNTunner: Clase que haciendo uso de un objeto de la clase BasicTimeSeriesANN permite probar
#       diferentes arquitecturas de RNA secuenciales (MLP, RNN o CNN) para tratar de resolver un problema de
#       predicción multistep de una serie temporal.
#
#       - SeqToSeqANN: Clase idéntica a la clase BasicTimeSeriesANN pero que únicamente permite definir modelos Seq2Seq.
######################################################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout, Reshape, GlobalMaxPooling1D, \
    BatchNormalization, LSTMCell, Bidirectional, RNN, concatenate
from keras.models import Model
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint
from numpy.random import seed
from tensorflow import set_random_seed


class BasicTimeSeriesANN:

    # Constructor de la clase que declara e inicializa los atributos, los cuales son:
    #
    #   - dataframe: Pandas dataframe con la serie temporal que se utilizará para entrenar y validar la
    #   red neuronal. Contiene solo los valores de las variables predictoras y de la variable objetivo.
    #   - target: String con el nombre de la columna del conjunto de datos (dataframe) que contiene los valores
    #   de la variable objetivo.
    #   - lookback: Número de timesteps que retrocedemos hacia atrás a partir de un timestep concreto de la serie
    #   temporal a la hora de crear una secuencia para entrenar y validar la red neuronal.
    #   - delay: Número de timesteps después del último timestep de la secuencia de entrada para los que queremos
    #   que la red neuronal prediga la variable objetivo.
    #   - date_train_ini: Fecha inicial de los datos de entrenamiento.
    #   - date_train_end: Fecha final de los datos de entrenamiento.
    #   - date_val_ini: Fecha inicial de los datos de validación.
    #   - date_val_end: Fecha final de los datos de validación.
    #   - hidden_layers: Lista que en la casilla i-ésima contiene un diccionario de pares clave-valor con el tipo
    #   y los parámetros de la capa oculta i-ésima. Los parámetros dependen del tipo de capa y para cada tipo el usuario
    #   debe proporcionar unos parámetros acordes con el mismo. A continuación se muestran los parámetros que se deben
    #   proporcionar para cada tipo de capa.
    #
    #
    #           - Dense:
    #
    #               - units: Número de unidades de la capa.
    #               - activation: String con la función de activación de cada unidad (las funciones se pueden consultar
    #               en la documentación de keras).
    #
    #           - Dropout:
    #
    #               - rate: Proporción de las unidades de entrada sobre las que aplicar el dropout.
    #
    #           - Flatten:
    #
    #               - Sin parámetros.
    #
    #           - Reshape:
    #
    #               - target_shape: Tupla de enteros con las nuevas dimensiones que tendrán los datos a la salida.
    #
    #           - Conv1D:
    #
    #               - filters: Número de filtros de convolución que se van a aplicar sobre los datos de entrada.
    #               - kernel_size: Tamaño de los filtros de convolución.
    #               - activation: String con la función de activación de cada unidad (las funciones se pueden consultar
    #               en la documentación de keras).
    #
    #           - LSTM:
    #
    #               - units: Número de unidades de la capa.
    #               - dropout: Proporción de las unidades que se quedarán fuera a la hora de llevar a cabo la
    #               transformación lineal de las entradas.
    #               - recurrent_dropout: Proporción de las unidades que se quedarán fuera a la hora de llevar a cabo la
    #               transformación lineal del estado recurrente.
    #               - return_sequences: Booleano que indica si devolvemos (True) o no (False) la secuencia completa.
    #
    #           - MaxPooling1D:
    #
    #               - pool_size: Tamaño de la "ventana" de pooling.
    #
    #           - GlobalMaxPooling:
    #
    #               - Sin parámetros.
    #
    #           - BatchNormalization:
    #
    #               - Sin parámetros.
    #
    #           - Ejemplo de uso:
    #
    #            - Definición de un red con una capa LSTM con 10 unidades seguida de una capa totalmente conectada con 5
    #            unidades.
    #
    #               hidden_layers = [{"type":"LSTM", "units":10, "dropout":0.5, "recurrent_dropout":0.30, return_sequences:False},
    #               {"type":"Dense", "units":5, "activation": "relu"}])
    #
    #   - optimizer: String con el nombre del algoritmo que se empleará para optimizar la función objetivo
    #   (de error). También puede ser un objeto de la clase keras.optimizers (consultar la documentación de
    #   Keras oara ver los distintos tipos de optimizadores).
    #   - loss: String con el nombre de la función objetivo, es decir, la función con la que mediremos el error
    #   cometido por la red neuronal. También puede ser una instancia de la clase keras.losses (consultar la
    #   documentación de Keras para ver los distintos tipos de funciones de pérdida (loss)).
    #   - metrics: Lista de Strings con las métricas de calidad que queremos calcular durante el entrenamiento y
    #   validación de la red neuronal.
    #   - logdir: Ruta al directorio en el cual queremos almacenar los logs generados por tensorboard relativos a
    #   la red neuronal y su entrenamiento.
    #   - distributed_train: Booleano que indica si el entrenamiento será distribuido (True) o no (False).
    #   Cuando el usuario decide el valor de los atributos, está definiendo el problema de predicción a resolver y la
    #   arquitectura y parámetros de la red neuronal que se empleará para afrontarlo.
    def __init__(self, dataframe, target, lookback, delay, hidden_layers, optimizer, loss, metrics,
                 logdir, distributed_train=False, date_train_ini=None, date_train_end=None, date_val_ini=None,
                 date_val_end=None):

        self._dataframe = dataframe
        self._target = target
        self._lookback = lookback
        self._delay = delay
        self._date_train_ini = date_train_ini
        self._date_train_end = date_train_end
        self._date_val_ini = date_val_ini
        self._date_val_end = date_val_end
        self._hidden_layers = hidden_layers
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        self._distributed_train = distributed_train

        # Definimos el callback tensorboard para poder monitorizar el entrenamiento con tensorflow.
        self._tensorboard = TensorBoard(log_dir=logdir)

        # Si el entrenamiento será distribuido normalizamos todo el conjunto de datos, en caso contrario solo
        # normalizamos los datos que se encuentran entre date_train_ini y date_val_end.
        if (distributed_train):

            # En primer lugar normalizamos el conjunto de datos.
            scaler = StandardScaler()
            norm_data = scaler.fit_transform(self._dataframe.values)
            self._norm_data = pd.DataFrame(norm_data, index=self._dataframe.index, columns=self._dataframe.columns)

        else:

            # Normalizamos los datos.
            scaler = StandardScaler()
            scaler.fit(self._dataframe.loc[self._date_train_ini:self._date_train_end].values)
            norm_data = scaler.transform(self._dataframe.loc[self._date_train_ini:self._date_val_end].values)
            self._norm_data = pd.DataFrame(data=norm_data,
                                           index=self._dataframe.loc[self._date_train_ini:self._date_val_end].index,
                                           columns=self._dataframe.loc[self._date_train_ini:self._date_val_end].columns)

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el conjunto de datos.
    def getDataFrame(self):

        return (self._dataframe)

    # Método para establecer el target.
    def setTarget(self, target):

        self._target = target

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el target.
    def getTarget(self):

        return (self._target)

    # Método para establecer el lookback.
    def setLookBack(self, lookback):

        self._lookback = lookback

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el lookback.
    def getLookBack(self):

        return (self._lookback)

    # Método para establecer el delay.
    def setDelay(self, delay):

        self._delay = delay

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

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

    # Método para consultar la variable distributed_train.
    def getDistributedTrain(self):

        return (self._distributed_train)

    # Método para establecer las capas ocultas.
    def setHiddenLayers(self, hidden_layers):

        self._hidden_layers = hidden_layers

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar las capas ocultas.
    def getHiddenLayers(self):

        return (self._hidden_layers)

    # Método para establecer el algoritmo de optimización.
    def setOptimizer(self, optimizer):

        self._optimizer = optimizer

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el algoritmo de optimización.
    def getOptimizer(self):

        return (self._optimizer)

    # Método para establecer la función de pérdida.
    def setLoss(self, loss):

        self._loss = loss

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar la función de pérdida.
    def getLoss(self):

        return (self._loss)

    # Método para establecer las métricas.
    def setMetrics(self, metrics):

        self._metrics = metrics

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar las métricas.
    def getMetrics(self):

        return (self._metrics)

    # Método para consultar el modelo.
    def getModel(self):

        return (self._model)

    # Método para consultar el historial de entrenamiento.
    def getHistory(self):

        return (self._history)

    # Método para consultar el tensorboard.
    def getTensorBoard(self):

        return (self._tensorboard)

    # Método para establecer el tensorboard..
    def setTensorBoard(self, tensorboard):

        self._tensorboard = tensorboard

    # Método que permite generar de manera indefinida lotes (batches) de muestras de entrenamiento a partir del
    # conjunto de datos de la clase. Dichas muestras se caracterizarán por los atributos lookback y
    # delay de la clase.
    #
    #   - Parámetros:
    #
    #       - min_date y max_date: Fechas en formato yyyy-mm-dd. Solo los timestep del conjunto de datos asociado
    #       a la clase se utilizarán para crear muestras.
    #       - batch_size: Es el número de muestras generadas en cada lote.
    #       - shuffle: Variable booleana que es True si queremos utilizar los timestep entre date_ini date_end en
    #       orden aleatorio para crear las muestras (ideal para el entrenamiento) y False en caso contrario
    #       (validación).
    #
    #   - Valor devuelto:
    #
    #       - sequences: Array tridimensional con el lote de batch_size muestras (series temporales).
    #       - targets: Array bidimensional o unidimensional (depende de si la predicción es multistep o unistep
    #       respectivamente) con los valores o valor que tiene que predecir la red neuronal al procesar cada una de
    #       las series temporales del lote.
    def train_generator(self, date_ini, date_end, batch_size, shuffle=False):

        # En primer lugar nos quedamos con los timesteps de nuestro conjunto de datos entre min_date y max_date.
        data = self._norm_data.loc[date_ini:date_end]

        # Ahora generamos la secuencia de índices de timesteps.
        indexes = np.arange(self._lookback, len(data) - self._delay)

        # Si shuffle es True barajamos los índices.
        if (shuffle):
            np.random.shuffle(indexes)

        # Contador total de secuencias creadas.
        n_samples = 0

        # Iteramos de manera indefinida y circular sobre los índices.
        while True:

            # Comprobar si podemos generar batch_size secuencias.
            if ((n_samples + batch_size) > len(indexes)):

                if (shuffle):
                    # Barajamos los índices de nuevo.
                    np.random.shuffle(indexes)

                n_samples = 0

            # Cogemos los siguientes batch_size índices.
            idx = indexes[n_samples:(n_samples + batch_size)]

            # Creamos los arrays en los que almacenar el lote secuencias.
            sequences = np.zeros(shape=(batch_size, self._lookback + 1, data.shape[-1]))

            # Creamos el array en el que almacenar los valores de la variable objetivo.
            targets = np.zeros(shape=(batch_size, self._delay))

            # Vamos creando el lote de secuencias con los siguientes batch_size timesteps.
            for j, i in enumerate(idx):
                # Creamos la secuencia.
                seq = data.iloc[np.arange(i - self._lookback, i + 1)].values

                # Creamos los targets de la secuencia actual.
                targ = data.iloc[np.arange(i + 1, i + self._delay + 1)][self._target].values

                sequences[j] = seq
                targets[j] = targ

            # Sumamos las batch_size muestras generadas.
            n_samples += batch_size

            # Generamos el lote de secuencias y los valores de la variable objetivo.
            yield sequences, targets

    # Método que permite generar de manera indefinida lotes (batches) de muestras de validación a partir del
    # conjunto de datos de la clase. Dichas muestras se caracterizarán por los atributos lookback y
    # delay de la clase.
    #
    #   - Parámetros:
    #
    #       - min_date y max_date: Fechas en formato yyyy-mm-dd. Solo los timestep del conjunto de datos asociado
    #       a la clase se utilizarán para crear muestras.
    #       - batch_size: Es el número de muestras generadas en cada lote.
    #       - shuffle: Variable booleana que es True si queremos utilizar los timestep entre date_ini date_end en
    #       orden aleatorio para crear las muestras y False en caso contrario.
    #
    #   - Valor devuelto:
    #
    #       - sequences: Array tridimensional con el lote de batch_size muestras (series temporales).
    #       - targets: Array bidimensional o unidimensional (depende de si la predicción es multistep o unistep
    #       respectivamente) con los valores o valor que tiene que predecir la red neuronal al procesar cada una de
    #       las series temporales del lote.
    def val_generator(self, date_ini, date_end, batch_size, shuffle=False):

        # En primer lugar nos quedamos con los timesteps de nuestro conjunto de datos entre min_date y max_date.
        data = self._norm_data.iloc[(self._norm_data.index.get_loc(date_ini)-self._lookback):self._norm_data.index.get_loc(date_end)]

        # Ahora generamos la secuencia de índices de timesteps.
        indexes = np.arange(self._lookback, len(data) - self._delay)

        # Si shuffle es True barajamos los índices.
        if (shuffle):
            np.random.shuffle(indexes)

        # Contador total de secuencias creadas.
        n_samples = 0

        # Iteramos de manera indefinida y circular sobre los índices.
        while True:

            # Comprobar si podemos generar batch_size secuencias.
            if ((n_samples + batch_size) > len(indexes)):

                if (shuffle):
                    # Barajamos los índices de nuevo.
                    np.random.shuffle(indexes)

                n_samples = 0

            # Cogemos los siguientes batch_size índices.
            idx = indexes[n_samples:(n_samples + batch_size)]

            # Creamos los arrays en los que almacenar el lote secuencias.
            sequences = np.zeros(shape=(batch_size, self._lookback + 1, data.shape[-1]))

            # Creamos el array en el que almacenar los valores de la variable objetivo.
            targets = np.zeros(shape=(batch_size, self._delay))

            # Vamos creando el lote de secuencias con los siguientes batch_size timesteps.
            for j, i in enumerate(idx):
                # Creamos la secuencia.
                seq = data.iloc[np.arange(i - self._lookback, i + 1)].values

                # Creamos los targets de la secuencia actual.
                targ = data.iloc[np.arange(i + 1, i + self._delay + 1)][self._target].values

                sequences[j] = seq
                targets[j] = targ

            # Sumamos las batch_size muestras generadas.
            n_samples += batch_size

            # Generamos el lote de secuencias y los valores de la variable objetivo.
            yield sequences, targets

    # Método que permite definir la arquitectura de la red neuronal a entrenar y validar a partir de las
    # capas ocultas especificadas por el usuario en el atributo hidden_layers. Las capas de entrada y salida se crean
    # a partir de los valores de los atributos lookback y delay.
    def defineArquitecture(self):

        # En primer lugar definimos la capa de entrada.
        input_layer = Input(shape=(self._lookback + 1, self._dataframe.shape[-1]))

        # La capa anterior es al inicio la capa de entrada.
        prev_layer = input_layer

        # Ahora recorremos y definimos las capas ocultas.
        for h in self._hidden_layers:

            # Variable en la que almacenamos la capa actual.
            act_layer = None

            try:

                # Comprobamos el tipo de capa.
                if (h["type"] == "Dense"):

                    # Definimos una capa totalmente conectada.
                    act_layer = Dense(units=h["units"], activation=h["activation"])(prev_layer)

                elif (h["type"] == "Dropout"):

                    # Definimos una capa de dropout.
                    act_layer = Dropout(rate=h["rate"])(prev_layer)

                elif (h["type"] == "Flatten"):

                    # Definimos una capa Flatten.
                    act_layer = Flatten()(prev_layer)

                elif (h["type"] == "Reshape"):

                    # Definimos una capa Reshape.
                    act_layer = Reshape(target_shape=h["target_shape"])(prev_layer)

                elif (h["type"] == "Conv1D"):

                    # Definimos una capa de convolución 1D.
                    act_layer = Conv1D(filters=h["filters"], kernel_size=h["kernel_size"], activation=h["activation"])(
                        prev_layer)

                elif (h["type"] == "LSTM"):

                    # Definimos una capa recurrente LSTM.
                    act_layer = LSTM(units=h["units"],
                                     dropout=h["dropout"], recurrent_dropout=h["recurrent_dropout"],
                                     return_sequences=h["return_sequences"])(prev_layer)

                elif (h["type"] == "MaxPooling1D"):

                    # Definimos una capa de pooling 1D.
                    act_layer = MaxPooling1D(pool_size=h["pool_size"])(prev_layer)

                elif (h["type"] == "GlobalMaxPooling1D"):

                    # Definimos una capa de global max pooling 1D.
                    act_layer = GlobalMaxPooling1D()(prev_layer)

                elif (h["type"] == "BatchNormalization"):

                    # Definimos una capa de normalización..
                    act_layer = BatchNormalization()(prev_layer)

                else:

                    # Lanzamos una excepción, pues no se ha proporcionado un tipo de capa válido.
                    raise WrongLayerException

            except WrongLayerException:

                print("El tipo de capa " + h["type"] + " no es correcto")
                return (-1)

            # Para la siguiente iteración la capa previa es la actual.
            prev_layer = act_layer

        output_layer = None

        # Definimos la capa de salida.
        output_layer = Dense(units=self._delay)(prev_layer)

        # Finalmente definimos el modelo.
        self._model = Model(inputs=input_layer, outputs=output_layer)

    # Método que permite establecer los parámetros de aprendizaje de la red neuronal preparándola para entrenar.
    def compile(self):

        # Compilamos el modelo con los parámetros recibidos.
        self._model.compile(self._optimizer, self._loss, self._metrics)

    # Método para entrenar la red neuronal.
    #
    #   - Parámetros:
    #
    #       - batch_train_size: Tamaño de los lotes de entrenamiento.
    #       - train_shuffle: Booleano que indica si generamos los lotes de entrenamiento en orden aleatorio (True) o
    #       no (False).
    #       - batch_val_size: Tamaño de los lotes de validación.
    #       - val_shuffle: Booleano que indica si generamos los lotes de validación en orden aleatorio (True) o
    #       no (False).
    #       - epochs: Épocas de entrenamiento de la red neuronal.
    #       - checkpoint: Booleano que indica si queremos almacenar en disco los pesos del modelo con menor error de
    #       validación.
    #       - checkpoint_dir: Directorio donde almacenar los pesos del mejor modelo, en caso de querer hacerlo.
    def train(self, batch_train_size, train_shuffle, batch_val_size, val_shuffle, epochs, checkpoint=False,
              checkpoint_path=None):

        # Si el entrenamiento es distribuido lanzamos una excepción.
        if (self._distributed_train):

            raise DistributedTrainException

        else:

            # Calculamos los pasos que tenemos que dar en cada época para poder recorrer todos los ejemplos de
            # entrenamiento y validación.
            train_steps = (len(self._norm_data.loc[
                               self._date_train_ini:self._date_train_end]) - self._lookback - self._delay) // batch_train_size
            val_steps = (len(self._norm_data.loc[
                             self._date_val_ini:self._date_val_end]) - self._delay) // batch_val_size

            # Ahora creamos los generadores de entrenamiento y validación.
            train_gen = self.train_generator(self._date_train_ini, self._date_train_end, batch_train_size, train_shuffle)
            val_gen = self.val_generator(self._date_val_ini, self._date_val_end, batch_val_size, val_shuffle)

            # Finalmente entrenamos el modelo.
            if (checkpoint):

                # Definimos el callback para hacer checkpoint del mejor modelo.
                check = ModelCheckpoint(checkpoint_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
                self._history = self._model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                                                          validation_data=val_gen, validation_steps=val_steps,
                                                          callbacks=[self._tensorboard, check])

            else:

                self._history = self._model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                                                          validation_data=val_gen, validation_steps=val_steps,
                                                          callbacks=[self._tensorboard])

    # Método para entrenar el modelo mediante generación distribuida de las muestras de entrenamiento y validación.
    # Para emplear este método es necesario utilizar todo el conjunto de datos.
    #
    #   - Parámetros:
    #
    #       - val_indexes: Índices de los timesteps del conjunto de datos asociado a esta clase a partir de los cuales
    #       vamos a crear las muestras de validación.
    #       - batch_size: Tamaño de los lotes de entrenamiento.
    #       - epochs: Épocas de entrenamiento de la red neuronal.
    #       - checkpoint: Booleano que indica si queremos almacenar en disco los pesos del modelo con menor error de
    #       validación.
    #       - checkpoint_dir: Directorio donde almacenar los pesos del mejor modelo, en caso de querer hacerlo.
    def train_distributed_data(self, val_indexes, batch_size, epochs, checkpoint=False, checkpoint_path=None):

        # Si el entrenamiento es distribuido lanzamos una excepción.
        if (not self._distributed_train):

            raise ContinuousTrainException

        else:

            # Creamos las muestras de validación y entrenamiento.
            val_x = []
            val_y = []

            idx_val_samples = set()

            # Recorremos los índices de los timesteps de validación y creamos una muestra con cada uno de ellos.
            for idx in val_indexes:
                val_sample_indexes = np.arange(idx - self._lookback, idx + 1, 1)

                # Creamos la muestra de validación.
                val_x.append(self._norm_data.iloc[val_sample_indexes].values)
                val_y.append(self._norm_data[self._target].iloc[np.arange(idx + 1, idx + 1 + self._delay, 1)].values)

                # Añadimos los índices de los timesteps de la muestra al conjunto.
                idx_val_samples = idx_val_samples.union(val_sample_indexes)

            val_x = np.array(val_x)
            val_y = np.array(val_y)

            # Ahora creamos las muestras de entrenamiento con el resto de timesteps.
            indexes = np.arange(self._lookback, len(self._norm_data) - self._delay, 1)

            train_x = []
            train_y = []

            for idx in indexes:

                # Creamos una muestra de entrenamiento con el índice actual.
                train_sample_indexes = np.arange(idx - self._lookback, idx + 1, 1)

                # Comprobamos si alguno de los índices se solapa con alguna muestra de validación.
                if (len(idx_val_samples.intersection(train_sample_indexes)) == 0):
                    train_x.append(self._norm_data.iloc[train_sample_indexes].values)
                    train_y.append(
                        self._norm_data[self._target].iloc[np.arange(idx + 1, idx + 1 + self._delay, 1)].values)

            train_x = np.array(train_x)
            train_y = np.array(train_y)

            # Comprobar si se desea hacer checkpoint.
            if (checkpoint):

                check = ModelCheckpoint(checkpoint_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
                self._history = self._model.fit(x=train_x,
                                                y=train_y,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                callbacks=[self._tensorboard, check],
                                                validation_data=(val_x, val_y)
                                                )
            else:

                self._history = self._model.fit(x=train_x,
                                                y=train_y,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                callbacks=[self._tensorboard],
                                                validation_data=(val_x, val_y)
                                                )

    # Método que pinta en dos gráficas la evolución de una determinada métrica sobre el conjunto de entrenamiento
    # y de validación a lo largo de las épocas de entrenamiento de la red neuronal.
    #
    #   - Parámetros:
    #
    #       - metric: Métrica cuya evolución queremos observar.
    def plotHistory(self, metric, title):

        # Obtenemos los valores de la métrica sobre el conjunto de entrenamiento y de validación.
        train_values = self._history.history[metric]
        val_values = self._history.history["val_" + metric]

        epochs = range(1, len(train_values) + 1)

        # Pintar.
        plt.figure(dpi=500)
        plt.style.use("seaborn-darkgrid")

        plt.plot(epochs, train_values, 'bo', label="Entrenamiento")
        plt.plot(epochs, val_values, 'b', label="Validación")
        plt.title(title)
        plt.xlabel("Épocas")
        plt.ylabel("NMAE")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

        plt.show()

    # Función para cargar en el modelo actual los pesos de un modelo almacenado en disco. El modelo almacenado en disco
    # y el actual deben tener la misma arquitectura e hiperparámetros.
    def loadWeigths(self, filepath):

        self._model.load_weights(filepath)
        self.compile()

    # Función que genera una serie temporal de longitud lookback a partir de un timestep concreto recibido como
    # argumento y a parir de ella predice los siguientes delay valores de la variable objetivo (target) con el modelo
    # actual, pintando estos últimos en una gráfica junto con los valores verdaderos.
    #
    #   - Parámetros:
    #
    #       - timestep: String con el instante de tiempo del conjunto de datos dataframe asociado al objeto de esta
    #       clase a partir del cual vamos a crear la serie temporal a predecir. Formato: yyyy-mm-dd HH:MM:SS.
    #       - title: Título de la gráfica.
    def plot_prediction_half_day(self, timestep, title):

        # Creamos los arrays en los que almacenar el lote secuencias.
        sequence = np.zeros(shape=(1, self._lookback + 1, self._norm_data.shape[-1]))

        # Índice del timestep.
        i = self._norm_data.index.get_loc(timestep)

        # Creamos la secuencia de entrada.
        sequence[0] = self._norm_data.iloc[np.arange(i - self._lookback, i + 1)].values

        # Creamos la secuencia de salida deseada.
        targets = self._norm_data[self._target].iloc[np.arange(i + 1, i + 1 + self._delay)].values

        # Ahora realizamos la predicción.
        prediction = self._model.predict(sequence)[0]

        # Pintamos los valores verdaderos y los predichos por el modelo.
        plt.figure(dpi=500)
        plt.style.use("seaborn-darkgrid")

        plt.plot(self._norm_data.index[np.arange(i + 1, i + 1 + self._delay)], prediction)
        plt.plot(self._norm_data.index[np.arange(i + 1, i + 1 + self._delay)], targets)

        plt.title(title)
        plt.xlabel("Tiempo")
        plt.ylabel(self._target + " Consumo")
        plt.legend(["Predicción", "Verdadero"], bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

        plt.show()

    # Método que predice con el modelo actual todos los valores de validación de la variable objetivo y los pinta
    # en una gráfica junto a los valores verdaderos.
    #
    #   - Parámetros:
    #
    #       - title: Título de la gráfica.
    def plot_prediction(self, title):

        if (self._distributed_train):

            raise DistributedTrainException

        else:

            # Generamos los índices de los timesteps del conjunto de datos que pertenecen al conjunto de validación y
            # que son susceptibles de crear muestras de validación.
            data = self._norm_data.iloc[(self._norm_data.index.get_loc(self._date_val_ini)-self._lookback):self._norm_data.index.get_loc(self._date_val_end)]
            indexes = np.arange(self._lookback, len(data) - self._delay, self._delay)

            # Listas en las que almacenar los valores predichos por el modelo y los verdaderos. Además de las fechas de
            # dichos datos.
            predicted_values = []
            true_values = []
            date_values = []

            for i in indexes:

                # Creamos la secuencia.
                seq = np.zeros(shape=(1, self._lookback + 1, self._norm_data.shape[-1]))
                seq[0] = data.iloc[np.arange(i - self._lookback, i + 1)].values
                targ = data.iloc[np.arange(i + 1, i + self._delay + 1)][self._target].values
                dates = data.iloc[np.arange(i + 1, i + self._delay + 1)].index.strftime("%Y-%m-%d %H:%M:%S").values

                # Predecimos los siguientes valores a partir de la muestra generada.
                pred = self._model.predict(seq)[0]

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


class WrongLayerException(Exception):
    pass


class DistributedTrainException(Exception):
    pass


class ContinuousTrainException(Exception):
    pass


class BasicTimeSeriesANNTunner:

    def __init__(self, dataframe, target, lookback, delay, loss, metrics,
                 logdir, date_train_ini, date_train_end, batch_train_size, train_shuffle, date_val_ini, date_val_end,
                 batch_val_size, val_shuffle, epochs):

        # Definimos el objeto de la clase BasicTimeSeriesAnn que albergará las múltiples arquitecturas a probar.
        self._ann = BasicTimeSeriesANN(dataframe=dataframe,
                                       target=target,
                                       lookback=lookback,
                                       delay=delay,
                                       date_train_ini=date_train_ini,
                                       date_train_end=date_train_end,
                                       date_val_ini=date_val_ini,
                                       date_val_end=date_val_end,
                                       hidden_layers=[],
                                       optimizer="adam",
                                       loss=loss,
                                       metrics=metrics,
                                       logdir="")

        # Definimos los atributos con los parámetros de entrenamiento.
        self._batch_train_size = batch_train_size
        self._train_shuffle = train_shuffle
        self._batch_val_size = batch_val_size
        self._val_shuffle = val_shuffle
        self._epochs = epochs
        self._logdir = logdir

    # Método que recibe como argumento una lista con una configuración de capas ocultas en cada casilla y otra con una
    # serie de funciones de optimización de modelos de redes neuronales y entrena un modelos de redes neuronales con
    # todas las combinaciones posibles de configuración de capas ocultas y función de optimización, almacenando
    # los resultados en un directorio de logs específico por cada modelo para que luego puedan ser visualizados mediante
    # Tensorboard.
    def tuneAnn(self, models_hidden_layers, optimizers):

        # Recorremos las configuraciones de capas ocultas.
        for hidden_layers in models_hidden_layers:

            for o in optimizers:
                # Definimos la arquitectura y la función de optimización del modelo actual.
                self._ann.setHiddenLayers(hidden_layers)
                self._ann.setOptimizer(o)
                self._ann.setTensorBoard(TensorBoard(log_dir=self._logdir + self.nameModel(hidden_layers) + o))

                # Cada modelo entrena siempre con la misma configuración aleatoria de pesos.
                seed(1)
                set_random_seed(2)

                # Entrenamos el modelo.
                self._ann.train(
                    batch_train_size=self._batch_train_size,
                    train_shuffle=self._train_shuffle,
                    batch_val_size=self._batch_val_size,
                    val_shuffle=self._val_shuffle,
                    epochs=self._epochs)

    # Función que genera el nombre de un modelo de red neuronal en función de sus capas ocultas.
    #
    #   - Parámetros:
    #
    #       - hidden_layers: Lista con las capas ocultas del modelo.
    #
    #   - Valor devuelto:
    #
    #       - model_name: Nombre del modelo generado a partir de sus capas ocultas.
    def nameModel(self, hidden_layers):

        # Creamos el nombre del modelo.
        model_name = ""

        for h in hidden_layers:

            model_name += h["type"][:4]

            try:

                # Comprobamos el tipo de capa.
                if (h["type"] == "Dense"):

                    model_name += str(h["units"])

                elif (h["type"] == "Dropout"):

                    model_name += str(h["rate"])

                elif (h["type"] == "Conv1D"):

                    model_name += str(h["filters"]) + "-" + str(h["kernel_size"])

                elif (h["type"] == "LSTM"):

                    model_name += str(h["units"]) + "-" + str(h["dropout"]) + "-" + str(h["recurrent_dropout"])

                elif (h["type"] == "MaxPooling1D"):

                    model_name += str(h["pool_size"])

                elif (h["type"] != "Reshape" and h["type"] != "Flatten" and h["type"] != "GlobalMaxPooling1D" and
                      h["type"] != "BatchNormalization"):

                    # Lanzamos una excepción, pues no se ha proporcionado un tipo de capa válido.
                    raise WrongLayerException

                model_name += "_"

            except WrongLayerException:

                print("El tipo de capa " + h["type"] + " no es correcto")
                return (-1)

        return (model_name)


class SeqToSeqANN:

    # Constructor de la clase que declara e inicializa los atributos, los cuales son:
    #
    #   - dataframe: Pandas dataframe con la serie temporal que se utilizará para entrenar y validar la
    #   red neuronal. Contiene solo los valores de las variables predictoras y de la variable objetivo.
    #   - target: String ocn el nombre de la columna del conjunto de datos (dataframe) que contiene los valores
    #   de la variable objetivo.
    #   - encoder_variables: Lista con los nombres de las variables del dataframe cuyos valores se utilizarán como
    #   entrada para el codificador de la red.
    #   - decoder_variables: Lista con los nombres de las variables del dataframe cuyos valores se utilizarán como
    #     entrada para el decodificador de la red.
    #   - lookback: Número de timesteps que retrocedemos hacia atrás a partir de un timestep concreto de la serie
    #   temporal a la hora de crear una secuencia para entrenar y validar la red neuronal.
    #   - delay: Número de timesteps después del último timestep de la secuencia de entrada para los que queremos
    #   que la red neuronal prediga la variable objetivo.
    #   - date_train_ini: Fecha inicial de los datos de entrenamiento.
    #   - date_train_end: Fecha final de los datos de entrenamiento.
    #   - date_val_ini: Fecha inicial de los datos de validación.
    #   - date_val_end: Fecha final de los datos de validación.
    #   - hidden_layers: Lista cuyo tamaño nos indica el número de capas LSTM del codificador y decodificador, y que
    #   en la casilla i contiene el número de neuronas de la capa LSTM i del codificador y del decodificador.
    #   - optimizer: String con el nombre del algoritmo que se empleará para optimizar la función objetivo
    #   (de error). También puede ser un objeto de la clase keras.optimizers (consultar la documentación de
    #   Keras oara ver los distintos tipos de optimizadores).
    #   - loss: String con el nombre de la función objetivo, es decir, la función con la que mediremos el error
    #   cometido por la red neuronal. También puede ser una instancia de la clase keras.losses (consultar la
    #   documentación de Keras para ver los distintos tipos de funciones de pérdida (loss)).
    #   - metrics: Lista de Strings con las métricas de calidad que queremos calcular durante el entrenamiento y
    #   validación de la red neuronal.
    #   - bidirectional: Booleano que indica si queremos que las capas LSTM del codificador y del decodificador sean
    #   bidireccionales.
    #   - logdir: Ruta al directorio en el cual queremos almacenar los logs generados por tensorboard relativos a
    #   la red neuronal y su entrenamiento.
    #   - distributed_train: Booleano que indica si el entrenamiento será distribuido (True) o no (False).
    #   Cuando el usuario decide el valor de los atributos, está definiendo el problema de predicción a resolver y la
    #   arquitectura y parámetros de la red neuronal que se empleará para afrontarlo.
    def __init__(self, dataframe, target, encoder_variables, decoder_variables, lookback, delay,
                 hidden_layers, optimizer, loss, metrics, bidirectional, logdir, distributed_train=False,
                 date_train_ini=None, date_train_end=None, date_val_ini=None, date_val_end=None):

        self._dataframe = dataframe
        self._target = target
        self._encoder_variables = encoder_variables
        self._decoder_variables = decoder_variables
        self._lookback = lookback
        self._delay = delay
        self._date_train_ini = date_train_ini
        self._date_train_end = date_train_end
        self._date_val_ini = date_val_ini
        self._date_val_end = date_val_end
        self._distributed_train = distributed_train
        self._hidden_layers = hidden_layers
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        self._bidirectional = bidirectional

        # Si el entrenamiento será distribuido normalizamos todo el conjunto de datos, en caso contrario solo
        # normalizamos los datos que se encuentran entre date_train_ini y date_val_end.
        if (distributed_train):

            # En primer lugar normalizamos el conjunto de datos.
            scaler = StandardScaler()
            norm_data = scaler.fit_transform(self._dataframe.values)
            self._norm_data = pd.DataFrame(norm_data, index=self._dataframe.index, columns=self._dataframe.columns)

        else:

            # Normalizamos los datos.
            scaler = StandardScaler()
            scaler.fit(self._dataframe.loc[self._date_train_ini:self._date_train_end].values)
            norm_data = scaler.transform(self._dataframe.loc[self._date_train_ini:self._date_val_end].values)
            self._norm_data = pd.DataFrame(data=norm_data,
                                           index=self._dataframe.loc[self._date_train_ini:self._date_val_end].index,
                                           columns=self._dataframe.loc[self._date_train_ini:self._date_val_end].columns)

        # Definimos el callback tensorboard para poder monitorizar el entrenamiento con tensorflow.
        self._tensorboard = TensorBoard(log_dir=logdir)

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el conjunto de datos.
    def getDataFrame(self):

        return (self._dataframe)

        # Método para establecer el target.

    def setTarget(self, target):

        self._target = target

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

        # Método para consultar el target.

    def getTarget(self):

        return (self._target)

    # Método para establecer las variables del codificador.
    def setEncoderVariables(self, encoder_variables):

        self._encoder_variables = encoder_variables

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar las variables del codificador.
    def getEncoderVariables(self):

        return (self._encoder_variables)

    # Método para establecer las variables del decodificador.
    def setDecoderVariables(self, decoder_variables):

        self._decoder_variables = decoder_variables

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar las variables del decodificador.
    def getDecoderVariables(self):

        return (self._decoder_variables)

    # Método para establecer el lookback.
    def setLookBack(self, lookback):

        self._lookback = lookback

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el lookback.
    def getLookBack(self):

        return (self._lookback)

    # Método para establecer el delay.
    def setDelay(self, delay):

        self._delay = delay

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el delay.
    def getDelay(self):

        return (self._delay)

    # Método para establecer el bidirectional.
    def setBidirectional(self, bidirectional):

        self._bidirectional = bidirectional

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el bidirectional.
    def getBidirectional(self):

        return (self._bidirectional)

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

    # Método para consultar la variable distributed_train.
    def getDistributedTrain(self):

        return (self._distributed_train)

    # Método para establecer las capas ocultas.
    def setHiddenLayers(self, hidden_layers):

        self._hidden_layers = hidden_layers

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar las capas ocultas.
    def getHiddenLayers(self):

        return (self._hidden_layers)

    # Método para establecer el algoritmo de optimización.
    def setOptimizer(self, optimizer):

        self._optimizer = optimizer

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar el algoritmo de optimización.
    def getOptimizer(self):

        return (self._optimizer)

    # Método para establecer la función de pérdida.
    def setLoss(self, loss):

        self._loss = loss

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar la función de pérdida.
    def getLoss(self):

        return (self._loss)

    # Método para establecer las métricas.
    def setMetrics(self, metrics):

        self._metrics = metrics

        # Definimos y compilamos la arquitectura.
        self.defineArquitecture()
        self.compile()

    # Método para consultar las métricas.
    def getMetrics(self):

        return (self._metrics)

    # Método para consultar el modelo.
    def getModel(self):

        return (self._model)

    # Método para consultar el historial de entrenamiento.
    def getHistory(self):

        return (self._history)

    # Método para consultar el tensorboard.
    def getTensorBoard(self):

        return (self._tensorboard)

    # Método para establecer el tensorboard.
    def setTensorBoard(self, tensorboard):

        self._tensorboard = tensorboard

    # Método que permite generar de manera indefinida lotes (batches) de muestras de entrenamiento
    # a partir del conjunto de datos de la clase. Dichas muestras se caracterizarán por los atributos lookback y
    # delay de la clase.
    #
    #   - Parámetros:
    #
    #       - min_date y max_date: Fechas en formato yyyy-mm-dd. Solo los timestep del conjunto de datos asociado
    #       a la clase se utilizarán para crear muestras.
    #       - batch_size: Es el número de muestras generadas en cada lote.
    #       - shuffle: Variable booleana que es True si queremos utilizar los timestep entre date_ini y date_end en
    #       orden aleatorio para crear las muestras y False en caso contrario.
    #
    #   - Valor devuelto:
    #
    #       - sequences: Array tridimensional con el lote de batch_size muestras (series temporales).
    #       - targets: Array bidimensional  con los valores o valor que tiene que predecir la red neuronal
    #       al procesar cada una de las series temporales del lote.
    def train_generator(self, date_ini, date_end, batch_size, shuffle=False):

        # En primer lugar nos quedamos con los timesteps de nuestro conjunto de datos entre date_ini y date_end.
        data = self._norm_data.loc[date_ini:date_end]

        # Ahora generamos la secuencia de índices de timesteps.
        indexes = np.arange(self._lookback, len(data) - self._delay)

        # Si shuffle es True barajamos los índices.
        if (shuffle):
            np.random.shuffle(indexes)

        # Contador total de secuencias creadas.
        n_samples = 0

        # Iteramos de manera indefinida y circular sobre los índices.
        while True:

            # Comprobar si podemos generar batch_size secuencias.
            if ((n_samples + batch_size) > len(indexes)):

                if (shuffle):
                    # Barajamos los índices de nuevo.
                    np.random.shuffle(indexes)

                n_samples = 0

            # Cogemos los siguientes batch_size índices.
            idx = indexes[n_samples:(n_samples + batch_size)]

            # Creamos los arrays en los que almacenar el lote secuencias.
            encoder_sequences = np.zeros(shape=(batch_size, self._lookback + 1, len(self._encoder_variables)))

            # Creamos el array en el que almacenar la entrada del decodificador.
            decoder_sequences = np.zeros(shape=(batch_size, self._delay, len(self._decoder_variables)))

            # Creamos el array en el que almacenar la salida del decodificador.
            targets = np.zeros(shape=(batch_size, self._delay))

            # Vamos creando el lote de secuencias con los siguientes batch_size timesteps.
            for j, i in enumerate(idx):
                # Creamos la secuencia del codificador.
                encoder_seq = data[self._encoder_variables].iloc[np.arange(i - self._lookback, i + 1)].values

                # Creamos la secuencia del decodificador.
                decoder_seq = data[self._decoder_variables].iloc[np.arange(i + 1, i + 1 + self._delay)].values

                # Creamos la secuencia de salida deseada.
                targ = data[self._target].iloc[np.arange(i + 1, i + self._delay + 1)].values

                encoder_sequences[j] = encoder_seq
                decoder_sequences[j] = decoder_seq
                targets[j] = targ

            # Añadimos las batch_size muestras generadas.
            n_samples += batch_size

            # Generamos el lote de secuencias y los valores de la variable objetivo.
            yield ([encoder_sequences, decoder_sequences], targets)

    # Método que permite generar de manera indefinida lotes (batches) de muestras de validación
    # a partir del conjunto de datos de la clase. Dichas muestras se caracterizarán por los atributos lookback y
    # delay de la clase.
    #
    #   - Parámetros:
    #
    #       - min_date y max_date: Fechas en formato yyyy-mm-dd. Solo los timestep del conjunto de datos asociado
    #       a la clase se utilizarán para crear muestras.
    #       - batch_size: Es el número de muestras generadas en cada lote.
    #       - shuffle: Variable booleana que es True si queremos utilizar los timestep entre date_ini y date_end en
    #       orden aleatorio para crear las muestras y False en caso contrario.
    #
    #   - Valor devuelto:
    #
    #       - sequences: Array tridimensional con el lote de batch_size muestras (series temporales).
    #       - targets: Array bidimensional  con los valores o valor que tiene que predecir la red neuronal
    #       al procesar cada una de las series temporales del lote.
    def val_generator(self, date_ini, date_end, batch_size, shuffle=False):

        # En primer lugar nos quedamos con los timesteps de nuestro conjunto de datos entre date_ini y date_end.
        data = self._norm_data.iloc[(self._norm_data.index.get_loc(date_ini) - self._lookback):self._norm_data.index.get_loc(date_end)]

        # Ahora generamos la secuencia de índices de timesteps.
        indexes = np.arange(self._lookback, len(data) - self._delay)

        # Si shuffle es True barajamos los índices.
        if (shuffle):
            np.random.shuffle(indexes)

        # Contador total de secuencias creadas.
        n_samples = 0

        # Iteramos de manera indefinida y circular sobre los índices.
        while True:

            # Comprobar si podemos generar batch_size secuencias.
            if ((n_samples + batch_size) > len(indexes)):

                if (shuffle):
                    # Barajamos los índices de nuevo.
                    np.random.shuffle(indexes)

                n_samples = 0

            # Cogemos los siguientes batch_size índices.
            idx = indexes[n_samples:(n_samples + batch_size)]

            # Creamos los arrays en los que almacenar el lote secuencias.
            encoder_sequences = np.zeros(shape=(batch_size, self._lookback + 1, len(self._encoder_variables)))

            # Creamos el array en el que almacenar la entrada del decodificador.
            decoder_sequences = np.zeros(shape=(batch_size, self._delay, len(self._decoder_variables)))

            # Creamos el array en el que almacenar la salida del decodificador.
            targets = np.zeros(shape=(batch_size, self._delay))

            # Vamos creando el lote de secuencias con los siguientes batch_size timesteps.
            for j, i in enumerate(idx):
                # Creamos la secuencia del codificador.
                encoder_seq = data[self._encoder_variables].iloc[np.arange(i - self._lookback, i + 1)].values

                # Creamos la secuencia del decodificador.
                decoder_seq = data[self._decoder_variables].iloc[np.arange(i + 1, i + 1 + self._delay)].values

                # Creamos la secuencia de salida deseada.
                targ = data[self._target].iloc[np.arange(i + 1, i + self._delay + 1)].values

                encoder_sequences[j] = encoder_seq
                decoder_sequences[j] = decoder_seq
                targets[j] = targ

            # Sumamos las batch_size muestras generadas.
            n_samples += batch_size

            # Generamos el lote de secuencias y los valores de la variable objetivo.
            yield ([encoder_sequences, decoder_sequences], targets)

    # Método que permite definir la arquitectura de la red neuronal sequence to sequence a entrenar y validar
    # a partir de las capas ocultas especificadas por el usuario en el atributo hidden_layers.
    def defineArquitecture(self):

        # Obtenemos el número de capas LSTM del codificador y del decodificador.
        n_layers = len(self._hidden_layers)

        encoder_inputs = Input(shape=(self._lookback + 1, len(self._encoder_variables)))

        # Definimos el codificador.
        lstm_cells = [LSTMCell(hidden_dim) for hidden_dim in self._hidden_layers]

        if (self._bidirectional):

            encoder = Bidirectional(RNN(lstm_cells, return_state=True))
            encoder_outputs_and_states = encoder(encoder_inputs)
            bi_encoder_states = encoder_outputs_and_states[1:]
            encoder_states = []

            for i in range(int(len(bi_encoder_states) / 2)):
                temp = concatenate([bi_encoder_states[i], bi_encoder_states[2 * n_layers + i]], axis=-1)
                encoder_states.append(temp)

        else:

            encoder = RNN(lstm_cells, return_state=True)
            encoder_outputs_and_states = encoder(encoder_inputs)
            encoder_states = encoder_outputs_and_states[1:]

        # Definimos el decodificador.
        decoder_inputs = Input(shape=(self._delay, len(self._decoder_variables)))

        if (self._bidirectional):

            decoder_cells = [LSTMCell(hidden_dim * 2) for hidden_dim in
                             self._hidden_layers]

        else:

            decoder_cells = [LSTMCell(hidden_dim) for hidden_dim in self._hidden_layers]

        decoder_lstm = RNN(decoder_cells, return_sequences=True, return_state=True)
        decoder_outputs_and_states = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_outputs = decoder_outputs_and_states[0]

        decoder_flatten = Flatten()
        decoder_flatten = decoder_flatten(decoder_outputs)
        decoder_dense = Dense(self._delay)
        decoder_dense = decoder_dense(decoder_flatten)

        # Definimos finalmente el modelo.
        self._model = Model([encoder_inputs, decoder_inputs], decoder_dense)

    # Método que permite establecer los parámetros de aprendizaje de la red neuronal preparándola para entrenar.
    def compile(self):

        # Compilamos el modelo con los parámetros recibidos.
        self._model.compile(self._optimizer, self._loss, self._metrics)

    # Método para entrenar la red neuronal.
    #
    #   - Parámetros:
    #       - batch_train_size: Tamaño de los lotes de entrenamiento.
    #       - train_shuffle: Booleano que indica si generamos los lotes de entrenamiento en orden aleatorio (True) o
    #       no (False).
    #       - batch_val_size: Tamaño de los lotes de validación.
    #       - val_shuffle: Booleano que indica si generamos los lotes de validación en orden aleatorio (True) o
    #       no (False).
    #       - epochs: Épocas de entrenamiento de la red neuronal.
    #       - checkpoint: Booleano que indica si queremos almacenar en disco los pesos del modelo con menor error de
    #       validación.
    #       - checkpoint_dir: Directorio donde almacenar los pesos del mejor modelo, en caso de querer hacerlo.
    def train(self, batch_train_size, train_shuffle, batch_val_size, val_shuffle, epochs, checkpoint=False,
              checkpoint_path=None):

        # Si el entrenamiento es distribuido lanzamos una excepción.
        if (self._distributed_train):

            raise DistributedTrainException

        else:

            # Calculamos los pasos que tenemos que dar en cada época para poder recorrer todos los ejemplos de
            # entrenamiento y validación.
            train_steps = (len(self._norm_data.loc[
                               self._date_train_ini:self._date_train_end]) - self._lookback - self._delay) // batch_train_size
            val_steps = (len(self._norm_data.loc[
                             self._date_val_ini:self._date_val_end]) - self._delay) // batch_val_size

            # Ahora creamos los generadores de entrenamiento y validación.
            train_gen = self.train_generator(self._date_train_ini, self._date_train_end, batch_train_size, train_shuffle)
            val_gen = self.val_generator(self._date_val_ini, self._date_val_end, batch_val_size, val_shuffle)

            # Finalmente entrenamos el modelo.

            if (checkpoint):

                # Definimos el callback para hacer checkpoint del mejor modelo.
                check = ModelCheckpoint(checkpoint_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
                self._history = self._model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                                                          validation_data=val_gen, validation_steps=val_steps,
                                                          callbacks=[self._tensorboard, check])

            else:

                self._history = self._model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                                                          validation_data=val_gen, validation_steps=val_steps,
                                                          callbacks=[self._tensorboard])

    # Método que pinta en dos gráficas la evolución de una determinada métrica sobre el conjunto de entrenamiento
    # y de validación a lo largo de las épocas de entrenamiento de la red neuronal.
    #
    #   - Parámetros:
    #
    #       - metric: Métrica cuya evolución queremos observar.
    def plotHistory(self, metric, title):

        # Obtenemos los valores de la métrica sobre el conjunto de entrenamiento y de validación.
        train_values = self._history.history[metric]
        val_values = self._history.history["val_" + metric]

        epochs = range(1, len(train_values) + 1)

        # Pintar.
        plt.figure(dpi=500)
        plt.style.use("seaborn-darkgrid")

        plt.plot(epochs, train_values, 'bo', label="Entrenamiento")
        plt.plot(epochs, val_values, 'b', label="Validación")
        plt.title(title)
        plt.xlabel("Épocas")
        plt.ylabel("NMAE")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

        plt.show()

    # Función para cargar en el modelo actual los pesos de un modelo almacenado en disco. El modelo almacenado en disco
    # y el actual deben tener la misma arquitectura e hiperparámetros.
    def loadWeigths(self, filepath):

        self._model.load_weights(filepath)
        self.compile()

    # Función que genera una serie temporal de longitud lookback a partir de un timestep concreto recibido como
    # argumento y a parir de ella predice los siguiente delay valores de la variable objetivo (target) con el modelo
    # actual, pintando estos últimos en una gráfica junto con los valores verdaderos.
    #
    #   - Parámetros:
    #
    #       - timestep: String con el instante de tiempo del conjunto de datos dataframe asociado al objeto de esta
    #       clase a partir del cual vamos a crear la serie temporal a predecir. Formato: yyyy-mm-dd HH:MM:SS.
    #       - title: Título de la gráfica.
    def plot_prediction_half_day(self, timestep, title):

        # Creamos los arrays en los que almacenar el lote secuencias.
        encoder_sequence = np.zeros(shape=(1, self._lookback + 1, len(self._encoder_variables)))

        # Creamos el array en el que almacenar la entrada del decodificador.
        decoder_sequence = np.zeros(shape=(1, self._delay, len(self._decoder_variables)))

        # Índice del timestep.
        i = self._norm_data.index.get_loc(timestep)

        # Creamos la secuencia del codificador.
        encoder_sequence[0] = self._norm_data[self._encoder_variables].iloc[np.arange(i - self._lookback, i + 1)].values

        # Creamos la secuencia del decodificador.
        decoder_sequence[0] = self._norm_data[self._decoder_variables].iloc[
            np.arange(i + 1, i + 1 + self._delay)].values

        # Creamos la secuencia de salida deseada.
        targets = self._norm_data[self._target].iloc[np.arange(i + 1, i + 1 + self._delay)].values

        # Ahora realizamos la predicción.
        prediction = self._model.predict([encoder_sequence, decoder_sequence])[0]

        # Pintamos los valores verdaderos y los predichos por el modelo.
        plt.figure(dpi=500)
        plt.style.use("seaborn-darkgrid")

        plt.plot(self._norm_data.index[np.arange(i + 1, i + 1 + self._delay)], prediction)
        plt.plot(self._norm_data.index[np.arange(i + 1, i + 1 + self._delay)], targets)

        plt.title(title)
        plt.xlabel("Tiempo")
        plt.ylabel(self._target + " Consumo")
        plt.legend(["Predicción", "Verdadero"], bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

        plt.show()

    # Método para entrenar el modelo mediante generación distribuida de las muestras de entrenamiento y validación.
    # Para emplear este método es necesario utilizar todo el conjunto de datos.
    #
    #   - Parámetros:
    #
    #       - val_indexes: Índices de los timesteps del conjunto de datos asociado a esta clase a partir de los cuales
    #       vamos a crear las muestras de validación.
    #       - batch_size: Tamaño de los lotes de entrenamiento.
    #       - epochs: Épocas de entrenamiento de la red neuronal.
    #       - checkpoint: Booleano que indica si queremos almacenar en disco los pesos del modelo con menor error de
    #       validación.
    #       - checkpoint_dir: Directorio donde almacenar los pesos del mejor modelo, en caso de querer hacerlo.
    def train_distributed_data(self, val_indexes, batch_size, epochs, checkpoint=False, checkpoint_path=None):

        # Si el entrenamiento es continuo lanzamos una excepción.
        if (not self._distributed_train):

            raise ContinuousTrainException

        else:

            # Creamos las muestras de validación y entrenamiento.
            val_x_encoder = []
            val_x_decoder = []
            val_y = []

            idx_val_samples = set()

            # Recorremos los índices de los timesteps de validación y creamos una muestra con cada uno de ellos.
            for idx in val_indexes:
                val_sample_indexes = np.arange(idx - self._lookback, idx + 1, 1)

                # Creamos la muestra de validación.
                val_x_encoder.append(self._norm_data[self._encoder_variables].iloc[val_sample_indexes].values)
                val_x_decoder.append(
                    self._norm_data[self._decoder_variables].iloc[np.arange(idx + 1, idx + 1 + self._delay, 1)].values)
                val_y.append(self._norm_data[self._target].iloc[np.arange(idx + 1, idx + 1 + self._delay, 1)].values)

                # Añadimos los índices de los timesteps de la muestra al conjunto.
                idx_val_samples = idx_val_samples.union(val_sample_indexes)

            val_x_encoder = np.array(val_x_encoder)
            val_x_decoder = np.array(val_x_decoder)
            val_y = np.array(val_y)

            # Ahora creamos las muestras de entrenamiento con el resto de timesteps.
            indexes = np.arange(self._lookback, len(self._norm_data) - self._delay, 1)

            train_x_encoder = []
            train_x_decoder = []
            train_y = []

            for idx in indexes:

                # Creamos una muestra de entrenamiento con el índice actual.
                train_sample_indexes = np.arange(idx - self._lookback, idx + 1, 1)

                # Comprobamos si alguno de los índices se solapa con alguna muestra de validación.
                if (len(idx_val_samples.intersection(train_sample_indexes)) == 0):
                    train_x_encoder.append(self._norm_data[self._encoder_variables].iloc[train_sample_indexes].values)
                    train_x_decoder.append(self._norm_data[self._decoder_variables].iloc[
                                               np.arange(idx + 1, idx + 1 + self._delay, 1)].values)
                    train_y.append(
                        self._norm_data[self._target].iloc[np.arange(idx + 1, idx + 1 + self._delay, 1)].values)

            train_x_encoder = np.array(train_x_encoder)
            train_x_decoder = np.array(train_x_decoder)
            train_y = np.array(train_y)

            print(len(train_x_encoder))

            # Comprobar si se desea hacer checkpoint.
            if (checkpoint):

                check = ModelCheckpoint(checkpoint_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
                self._history = self._model.fit(x=[train_x_encoder, train_x_decoder],
                                                y=train_y,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                callbacks=[self._tensorboard, check],
                                                validation_data=([val_x_encoder, val_x_decoder], val_y)
                                                )
            else:

                self._history = self._model.fit(x=[train_x_encoder, train_x_decoder],
                                                y=train_y,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                callbacks=[self._tensorboard],
                                                validation_data=([val_x_encoder, val_x_decoder], val_y)
                                                )

    # Método que predice con el modelo actual todos los valores de validación de la variable objetivo y los pinta
    # en una gráfica junto a los valores verdaderos.
    #
    #   - Parámetros:
    #
    #       - title: Título de la gráfica.
    def plot_prediction(self, title):

        if (self._distributed_train):

            raise DistributedTrainException

        else:

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
                seq_encoder = np.zeros(shape=(1, self._lookback + 1, len(self._encoder_variables)))
                seq_encoder[0] = data[self._encoder_variables].iloc[np.arange(i - self._lookback, i + 1)].values
                seq_decoder = np.zeros(shape=(1, self._delay, len(self._decoder_variables)))
                seq_decoder[0] = data[self._decoder_variables].iloc[np.arange(i + 1, i + self._delay + 1)].values

                targ = data.iloc[np.arange(i + 1, i + self._delay + 1)][self._target].values
                dates = data.iloc[np.arange(i + 1, i + self._delay + 1)].index.strftime("%Y-%m-%d %H:%M:%S").values

                # Predecimos los siguientes valores a partir de la muestra generada.
                pred = self._model.predict([seq_encoder, seq_decoder])[0]

                # Añadimos los valores predichos y los verdaderos a su corresondientes listas.
                predicted_values += pred.tolist()
                true_values += targ.tolist()
                date_values += dates.tolist()

            # Pintamos los valores verdaderos y los predichos por el modelo.
            fig = plt.figure(dpi=500)
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