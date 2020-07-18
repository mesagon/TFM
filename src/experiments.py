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
# - experiments.py: Script que contiene el código para ejecutar los experimentos dirigidos a
#  a la elección del mejor modelo de cada tipo (XGBoost, MLP, RNN, CNN y Seq2Seq) para cada problema
#  (enero,febrero y marzo). En el caso de los experimentos con los modelos XGBoost y Seq2Seq, los hiperparámetros con
#  los que hemos probado se encuentran en el código de este script pero los hiperparámetros con los que hemos probado
#  para los modelos MLP, RNN y CNN se encuentran en los siguientes ficheros ubicados en el directorio experiments:
#
#       - modelos_mlp: Contiene todas las configuraciones de capas ocultas que hemos probado para los modelos MLP
#       sobre los problemas de enero, febrero y marzo.
#
#       - modelos_rnn_enero: Contiene todas las configuraciones de capas ocultas que hemos probado para los modelos RNN
#       sobre el problema de enero.
#
#       - modelos_rnn_febrero: Contiene todas las configuraciones de capas ocultas que hemos probado para los modelos RNN
#       sobre el problema de febrero.
#
#       - modelos_rnn_marzo: Contiene todas las configuraciones de capas ocultas que hemos probado para los modelos RNN
#       sobre el problema de marzo.

#       - modelos_cnn_enero: Contiene todas las configuraciones de capas ocultas que hemos probado para los modelos CNN
#       sobre el problema de enero.
#
#       - modelos_cnn_febrero: Contiene todas las configuraciones de capas ocultas que hemos probado para los modelos CNN
#       sobre el problema de febrero.
#
#       - modelos_cnn_marzo: Contiene todas las configuraciones de capas ocultas que hemos probado para los modelos CNN
#       sobre el problema de marzo.
#
#  Cada uno de estos ficheros almacena las configuraciones en una lista en formato json, donde cada elemento es
#  una lista de diccionarios que representa una configuración de capas ocultas concreta (modelo de RNA), el cual en
#  la posición i-ésima contiene un diccionario de pares clave-valor que especifica
#  el tipo e hiperparámetros de la capa oculta i-iésima de dicha configuración. Cada uno de estos ficheros es leido y
#  parseado por el modelo BasicTimeSeriesANNTunner el cual prueba todas y cada una de las configuraciones de capas
#  ocultas del fichero sobre un problema concreto. Almacenamos estas configuraciones en disco para que el código de este
#  script no sea demasiado largo y confuso.
######################################################################################################################

import src.NeuronalNetworks
import src.XGBoost
import pickle
import json

if __name__ == "__main__":

    f = open("../doc/datasets/H_F123_D1_consumption.pkl", "rb")
    data_consumption_H_F123_D1 = pickle.load(f)
    f.close()

    # Experimento 0: Modelo XGBoost de regresion múltiple.

    # Probar varios modelos de XGBoost cada uno con distintos parámetros.
    n_estimators = [50, 100, 500]
    max_depth = [51, 100, 500]
    learning_rate = [0.05, 0.001, 0.1]

    # Problema de Enero.
    for n_est in n_estimators:

        for depth in max_depth:

            for learn in learning_rate:
                xgbTS = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D1,
                                                      target="H-F123-D1",
                                                      lookback=672,
                                                      delay=96,
                                                      date_train_ini="2016-01-05 00:15:00",
                                                      date_train_end="2016-01-21 23:45:00",
                                                      date_val_ini="2016-01-22 00:00:00",
                                                      date_val_end="2016-01-31 23:45:00",
                                                      n_estimators=n_est,
                                                      max_depth=depth,
                                                      learning_rate=learn)

                xgbTS.train()

                print(str(n_est) + "-" + str(depth) + "-" + str(learn) + " : " + str(xgbTS.evaluateModel()))

    # Problema de Febrero.
    for n_est in n_estimators:

        for depth in max_depth:

            for learn in learning_rate:
                xgbTS = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D1,
                                                      target="H-F123-D1",
                                                      lookback=672,
                                                      delay=96,
                                                      date_train_ini="2016-02-01 00:00:00",
                                                      date_train_end="2016-02-21 23:45:00",
                                                      date_val_ini="2016-02-22 00:00:00",
                                                      date_val_end="2016-02-29 23:45:00",
                                                      n_estimators=n_est,
                                                      max_depth=depth,
                                                      learning_rate=learn)

                xgbTS.train()

                print(str(n_est) + "-" + str(depth) + "-" + str(learn) + " : " + str(xgbTS.evaluateModel()))

    # Problema de Marzo.
    for n_est in n_estimators:

        for depth in max_depth:

            for learn in learning_rate:
                xgbTS = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D1,
                                                      target="H-F123-D1",
                                                      lookback=672,
                                                      delay=96,
                                                      date_train_ini="2016-03-01 00:00:00",
                                                      date_train_end="2016-03-21 23:45:00",
                                                      date_val_ini="2016-03-22 00:00:00",
                                                      date_val_end="2016-03-31 23:45:00",
                                                      n_estimators=n_est,
                                                      max_depth=depth,
                                                      learning_rate=learn)

                xgbTS.train()

                print(str(n_est) + "-" + str(depth) + "-" + str(learn) + " : " + str(xgbTS.evaluateModel()))

    # Experimento 1: Perceptrón multicapa (MLP).

    # Modelos de MLP con los que vamos a experimentar.
    f = open("../doc/experiments/modelos_mlp.txt", "r")
    models_hidden_layers = json.load(f)
    f.close()

    # Problema de enero.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/MLP/H_F123_D1/Enero/",
                                                             date_train_ini="2016-01-05 00:15:00",
                                                             date_train_end="2016-01-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-03-22 00:00:00",
                                                             date_val_end="2016-03-31 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["adam", "rmsprop", "sgd"])

    # Problema de febrero.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/MLP/H_F123_D1/Febrero/",
                                                             date_train_ini="2016-02-01 00:00:00",
                                                             date_train_end="2016-02-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-02-22 00:00:00",
                                                             date_val_end="2016-02-29 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["adam", "rmsprop", "sgd"])

    # Problema de marzo.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/MLP/H_F123_D1/Marzo/",
                                                             date_train_ini="2016-03-01 00:00:00",
                                                             date_train_end="2016-03-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-03-22 00:00:00",
                                                             date_val_end="2016-03-31 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["adam", "rmsprop", "sgd"])

    # Experimento 2: Redes neuronales recurrentes.

    # Problema de enero.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/RNN/H_F123_D1/Enero/",
                                                             date_train_ini="2016-01-05 00:15:00",
                                                             date_train_end="2016-01-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-01-22 00:00:00",
                                                             date_val_end="2016-01-31 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)


    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    f = open("../doc/experiments/modelos_rnn_enero.txt", "r")
    models_hidden_layers = json.load(f)
    f.close()

    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["rmsprop"])

    # Problema de febrero.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/RNN/H_F123_D1/Febrero/",
                                                             date_train_ini="2016-02-01 00:00:00",
                                                             date_train_end="2016-02-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-02-22 00:00:00",
                                                             date_val_end="2016-02-31 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    f = open("../doc/experiments/modelos_rnn_febrero.txt", "r")
    models_hidden_layers = json.load(f)
    f.close()

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["adam"])

    # Problema de marzo.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/RNN/H_F123_D1/Marzo/",
                                                             date_train_ini="2016-03-01 00:00:00",
                                                             date_train_end="2016-03-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-03-22 00:00:00",
                                                             date_val_end="2016-03-31 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    f = open("../doc/experiments/modelos_rnn_marzo.txt", "r")
    models_hidden_layers = json.load(f)
    f.close()

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["rmsprop"])

    # Experimento 3: Redes neuronales convolucionales recurrentes.

    # Problema de enero.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/CNN/H_F123_D1/Enero/",
                                                             date_train_ini="2016-01-05 00:15:00",
                                                             date_train_end="2016-01-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-01-22 00:00:00",
                                                             date_val_end="2016-01-31 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    f = open("../doc/experiments/modelos_cnn_enero.txt", "r")
    models_hidden_layers = json.load(f)
    f.close()

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["rmsprop"])

    # Problema de febrero.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/CNN/H_F123_D1/Febrero/",
                                                             date_train_ini="2016-02-01 00:00:00",
                                                             date_train_end="2016-02-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-02-22 00:00:00",
                                                             date_val_end="2016-02-29 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    f = open("../doc/experiments/modelos_cnn_febrero.txt", "r")
    models_hidden_layers = json.load(f)
    f.close()

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["adam"])

    # Problema de marzo.
    annTuner = src.NeuronalNetworks.BasicTimeSeriesANNTunner(dataframe=data_consumption_H_F123_D1,
                                                             target="H-F123-D1",
                                                             lookback=384,
                                                             delay=48,
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/CNN/H_F123_D1/Marzo/",
                                                             date_train_ini="2016-03-01 00:00:00",
                                                             date_train_end="2016-03-21 23:45:00",
                                                             batch_train_size=72,
                                                             train_shuffle=True,
                                                             date_val_ini="2016-03-22 00:00:00",
                                                             date_val_end="2016-03-31 23:45:00",
                                                             batch_val_size=76,
                                                             val_shuffle=False,
                                                             epochs=50)

    f = open("../doc/experiments/modelos_cnn_marzo.txt", "r")
    models_hidden_layers = json.load(f)
    f.close()

    # Probamos varias arquitecturas con diferentes hiperparámetros para resolver el problema.
    annTuner.tuneAnn(models_hidden_layers=models_hidden_layers, optimizers=["rmsprop"])

    # Experimento 4: Arquitectura sequence to sequence.

    # Problema de enero.

    # Modelos de una capa LSTM en el codificador y el decodificador.
    hidden_neurons = [[16], [32], [64], [128], [16, 16], [32, 32], [64, 64], [128, 128]]

    for h in hidden_neurons:

        seqToSeq = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D1,
                                                    target="H-F123-D1",
                                                    encoder_variables=["H-F123-D1", "H-F123-D5/2", "H-F1-D1-PAN",
                                                                       "H-PAW03", "H-PAW02"],
                                                    decoder_variables=["Occupation", "Ext-Temp"],
                                                    lookback=384,
                                                    delay=48,
                                                    date_train_ini="2016-01-05 00:15:00",
                                                    date_train_end="2016-01-21 23:45:00",
                                                    date_val_ini="2016-01-22 00:00:00",
                                                    date_val_end="2016-01-31 23:45:00",
                                                    hidden_layers=h,
                                                    optimizer="rmsprop",
                                                    loss="mae",
                                                    metrics=["mape"],
                                                    bidirectional=False,
                                                    logdir="../logs/SeqToSeq/H_F123_D1/Enero/" + str(h))

        seqToSeq.train(batch_train_size=72,
                       train_shuffle=True,
                       batch_val_size=76,
                       val_shuffle=False,
                       epochs=50)

    # Problema de febrero.
    for h in hidden_neurons:

        seqToSeq = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D1,
                                                    target="H-F123-D1",
                                                    encoder_variables=["H-F123-D1", "H-F123-D5/2", "H-F1-D1-PAN",
                                                                       "H-PAW03", "H-PAW02"],
                                                    decoder_variables=["Occupation", "Ext-Temp"],
                                                    lookback=384,
                                                    delay=48,
                                                    date_train_ini="2016-02-01 00:00:00",
                                                    date_train_end="2016-02-21 23:45:00",
                                                    date_val_ini="2016-02-22 00:00:00",
                                                    date_val_end="2016-02-29 23:45:00",
                                                    hidden_layers=h,
                                                    optimizer="adam",
                                                    loss="mae",
                                                    metrics=["mape"],
                                                    bidirectional=False,
                                                    logdir="../logs/SeqToSeq/H_F123_D1/Febrero/" + str(h))

        seqToSeq.train(batch_train_size=72,
                       train_shuffle=True,
                       batch_val_size=76,
                       val_shuffle=False,
                       epochs=50)

    # Problema de marzo.
    for h in hidden_neurons:

        seqToSeq = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D1,
                                                    target="H-F123-D1",
                                                    encoder_variables=["H-F123-D1", "H-F123-D5/2",
                                                                       "H-F1-D1-PAN",
                                                                       "H-PAW03", "H-PAW02"],
                                                    decoder_variables=["Occupation", "Ext-Temp"],
                                                    lookback=384,
                                                    delay=48,
                                                    date_train_ini="2016-03-01 00:00:00",
                                                    date_train_end="2016-03-21 23:45:00",
                                                    date_val_ini="2016-03-22 00:00:00",
                                                    date_val_end="2016-03-31 23:45:00",
                                                    hidden_layers=h,
                                                    optimizer="rmsprop",
                                                    loss="mae",
                                                    metrics=["mape"],
                                                    bidirectional=False,
                                                    logdir="../logs/SeqToSeq/H_F123_D1/Marzo/" + str(h))

        seqToSeq.train(batch_train_size=72,
                       train_shuffle=True,
                       batch_val_size=76,
                       val_shuffle=False,
                       epochs=50)