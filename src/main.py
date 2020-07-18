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
# - main.py: Script principal del proyecto que contiene el código desarrollado para llevar a cabo las siguientes tareas:
#
#       1.- Obtención de los valores de los sensores del edificio ICPE, haciendo uso de la clase ICPEDataExtractor
#       definida en el script data_extraction.py.
#
#       2.- Análisis exploratorio y preprocesamiento de los valores de los sensores obtenidos, empleando las funciones
#       del script data_analysis.py. Aunque las tareas de preprocesamiento de tratamiento de outliers valores perdidos
#       se lleva a cabo en R en el script preprocessing.R. Esto es debido a que el lenguaje R dispone de más herramientas
#       para abordar estas dos tareas.
#
#       3.- Comparación del rendimiento del mejor modelo de XGBoost, MLP, RNN, CNN y Seq2Seq para cada problema (enero,
#       febrero y marzo) obtenidos en el script experiments.py. Para ello, hace uso de las clases XGBoostTimeSeries del
#       script XGBoost.py y BasicTimeSeriesANN y Seq2SeqANN ambas del script NeuronalNetworks.py.
######################################################################################################################

from src.data_extraction import ICPEDataExtractor
import src.data_analysis
import src.NeuronalNetworks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import src.XGBoost
from fancyimpute import IterativeImputer
import os

if __name__ == "__main__":

    # Lista con los IDs de los sensores de la zona piloto.
    pilot_zone_sensors = [9015, 8971, 8978, 9020, 9021, 8868, 8937, 9024, 8865, 8933, 8929,
                          8886, 8951, 9093, 9092, 9091, 9074, 9086, 9059, 9085, 9090, 9073, 9094, 9049]

    # Ordenamos los sensores de menor a mayor.
    pilot_zone_sensors = sorted(pilot_zone_sensors)

    dataExtractor = ICPEDataExtractor(os.environ["USER"],os.environ["PASSWORD"])

    # Obtener los sensores que miden en la mayoría de días de los meses de enero, febrero, marzo y abril.
    jan_sensors = dataExtractor.everydaySensors("2016-01-05", "2016-01-31")
    feb_sensors = dataExtractor.everydaySensors("2016-02-01", "2016-02-25")
    march_sensors = dataExtractor.everydaySensors("2016-03-01", "2016-03-31")
    april_sensors = dataExtractor.everydaySensors("2016-04-01", "2016-04-25")

    # Obtener los sensores que miden todos los días de todos los meses.
    sensors = set.intersection(jan_sensors, feb_sensors, march_sensors, april_sensors)

    # Comprobar si los sensores de la zona piloto se encuentran entre los sensores que miden todos los días de todos los meses.
    print(set(pilot_zone_sensors).issubset(sensors))

    # Obtenemos los valores de los sensores en los meses de Enero, Febrero, Marzo y Abril.
    jan_sensors_values = dataExtractor.sensorsValues("2016-01-05", "2016-01-31")
    feb_sensors_values = dataExtractor.sensorsValues("2016-02-01", "2016-02-25")
    march_sensors_values = dataExtractor.sensorsValues("2016-03-01", "2016-03-31")
    april_sensors_values = dataExtractor.sensorsValues("2016-04-01", "2016-04-25")

    jan_sensors_values = jan_sensors_values[pilot_zone_sensors]
    feb_sensors_values = feb_sensors_values[pilot_zone_sensors]
    march_sensors_values = march_sensors_values[pilot_zone_sensors]
    april_sensors_values = april_sensors_values[pilot_zone_sensors]

    # Ponemos los pandas dataframe en uno solo.
    sensor_values_group_1 = pd.concat(
        [jan_sensors_values, feb_sensors_values, march_sensors_values, april_sensors_values])

    # Almacenamos el conjunto de datos en disco.
    f = open("../doc/datasets/sensor_values_group_1.pkl", "wb")
    pickle.dump(sensor_values_group_1, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # Leemos de disco el conjunto de datos con los valores de los sensores de la zona piloto
    # en enero, febrero, marzo y abril.
    f = open("../doc/datasets/sensor_values_group_1.pkl", "rb")
    sensor_values_group_1 = pickle.load(f)
    f.close()

    # Obtener los sensores que miden la mayoría de los días de los meses de Septiembre, Octubre y Noviembre.
    sept_sensors = dataExtractor.everydaySensors("2016-09-02", "2016-09-30")
    oct_sensors = dataExtractor.everydaySensors("2016-10-01", "2016-10-31")
    nov_sensors = dataExtractor.everydaySensors("2016-11-01", "2016-11-11")

    # Obtener los sensores que miden todos los días de todos los meses.
    sensors = set.intersection(sept_sensors, oct_sensors, nov_sensors)

    # Comprobar si los sensores de la zona piloto se encuentran entre los sensores que miden todos los días de todos los meses.
    print(set(pilot_zone_sensors).issubset(sensors))

    sept_sensors_values = dataExtractor.sensorsValues("2016-09-02", "2016-09-30")
    oct_sensors_values = dataExtractor.sensorsValues("2016-10-01", "2016-10-31")
    nov_sensors_values = dataExtractor.sensorsValues("2016-11-01", "2016-11-11")

    # Nos quedamos únicamnete con los valores de los sensores de la zona piloto.
    sept_sensors_values = sept_sensors_values[pilot_zone_sensors]
    oct_sensors_values = oct_sensors_values[pilot_zone_sensors]
    nov_sensors_values = nov_sensors_values[pilot_zone_sensors]

    # Ponemos los pandas dataframe en uno solo.
    sensor_values_group_2 = pd.concat([sept_sensors_values, oct_sensors_values, nov_sensors_values])

    # Almacenamos el conjunto de datos en disco.
    f = open("../doc/datasets/sensor_values_group_2.pkl", "wb")
    pickle.dump(sensor_values_group_2, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # Leemos de disco el conjunto de datos con los valores de los sensores de la zona piloto
    # en septiembre, octubre y noviembre.
    f = open("../doc/datasets/sensor_values_group_2.pkl", "rb")
    sensor_values_group_2 = pickle.load(f)
    f.close()

    # Introducimos en el primer conjunto de datos los valores perdidos.
    missing_values = pd.DataFrame(np.nan, index=pd.date_range(start="2016-02-26 00:00:00", end="2016-02-29 23:45:00",
                                                              freq="15min"), columns=sensor_values_group_1.columns,
                                  dtype=np.float64)
    sensor_values_group_1 = pd.concat(
        [sensor_values_group_1.loc["2016-01-05 00:00:00":"2016-02-25 23:45:00"], missing_values,
         sensor_values_group_1.loc["2016-03-01 00:00:00":"2016-04-25 23:45:00"]])

    # Ahora creamos una tabla con los porcentajes de ocupación en cada minuto del edificio icpe cada día
    # de la semana.
    weekly_occupation = pd.DataFrame(
        index=pd.date_range(start="00:00:00", end="23:59:00", freq="1min").strftime("%H:%M:%S"),
        columns=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], dtype=np.int64)

    # Rellenamos las ocupaciones del lunes.
    weekly_occupation.loc["00:00:00":"07:20:00", "Monday"] = 0.0
    weekly_occupation.loc["07:20:00":"07:30:00", "Monday"] = np.interp(range(11), [0, 10], [0, 100])
    weekly_occupation.loc["07:30:00":"12:00:00", "Monday"] = 100.0
    weekly_occupation.loc["12:00:00":"12:30:00", "Monday"] = np.interp(range(31), [0, 30], [100, 75])
    weekly_occupation.loc["12:30:00":"13:00:00", "Monday"] = np.interp(range(31), [0, 30], [75, 100])
    weekly_occupation.loc["13:00:00":"16:00:00", "Monday"] = 100.0
    weekly_occupation.loc["16:00:00":"16:10:00", "Monday"] = np.interp(range(11), [0, 10], [100, 0])
    weekly_occupation.loc["16:10:00":"23:59:00", "Monday"] = 0.0

    # La ocupación es la misma el lunes. martes, miércoles y jueves.
    weekly_occupation["Tuesday"] = weekly_occupation["Wednesday"] = weekly_occupation["Thursday"] = weekly_occupation[
        "Monday"]

    # El viernes es parecido a los dias anteriores con la diferencia de que la ocupación empieza a descender antes.
    weekly_occupation["Friday"] = weekly_occupation["Monday"]
    weekly_occupation.loc["13:30:00":"13:40:00", "Friday"] = np.interp(range(11), [0, 10], [100, 0])
    weekly_occupation.loc["13:40:00":"23:59:00", "Friday"] = 0.0

    # El sábado y el domingo no hay ocupación.
    weekly_occupation["Saturday"] = weekly_occupation["Sunday"] = 0.0

    # Añadimos la columna occupation.
    sensor_values_group_1 = src.data_analysis.add_occupation(sensor_values_group_1, weekly_occupation)
    sensor_values_group_2 = src.data_analysis.add_occupation(sensor_values_group_2, weekly_occupation)

    # Añadimos la variable temperatura externa.

    # Leemos el conjunto de datos que contiene las temperaturas.
    temp_data = pd.read_csv("../doc/datasets/ICPE_wheather_data.csv", sep=",", index_col="Timestamp", parse_dates=True)

    # Añadimos los valores de temperatura (columna SC00312) como una columna nueva en nuestro conjunto de datos.
    sensor_values_group_1["Ext-Temp"] = np.concatenate(
        [temp_data.loc["2017-01-04 22:00:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-27 22:00:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-28 22:00:00":"2017-04-25 21:45:00", "SC000312"].values])

    # Análisis exploratorio de los datos.

    # En primer lugar, vamos a cambiar el nombre de las columnas de ambos conjuntos de datos para poder identificarlas mejor.
    sensor_values_group_1 = sensor_values_group_1.rename(
        columns={8971: "E-F1-D5/2-C03", 8978: "E-F1-D5/2-C05", 9020: "E-F1-D5/2-C06", 9015: "E-F1-D1",
                 9021: "E-F1-D5/2-C08", 8868: "E-F2-D1", 9024: "E-F2-D5/2-C11", 8865: "E-F2-D5/2-C12", 8929: "E-F3-D1",
                 8933: "E-F2-D5/2-C15", 8937: "E-F2-D2", 8951: "E-F3-D5/2", 8886: "E-F3-D2", 9093: "H-F123-D1",
                 9091: "H-F123-D5/2E", 9074: "H-F123-D5/2W", 9086: "H-F1-D1-PAN", 9059: "H-F1-D5/2W-PAS",
                 9085: "H-F2-D5/2W-PAW", 9090: "H-PAW02",
                 9073: "H-PAW03", 9094: "H-PAW01", 9049: "W-ALL"})

    sensor_values_group_2 = sensor_values_group_2.rename(
        columns={8971: "E-F1-D5/2-C03", 8978: "E-F1-D5/2-C05", 9020: "E-F1-D5/2-C06", 9015: "E-F1-D1",
                 9021: "E-F1-D5/2-C08", 8868: "E-F2-D1", 9024: "E-F2-D5/2-C11", 8865: "E-F2-D5/2-C12", 8929: "E-F3-D1",
                 8933: "E-F2-D5/2-C15", 8937: "E-F2-D2", 8951: "E-F3-D5/2", 8886: "E-F3-D2", 9093: "H-F123-D1",
                 9091: "H-F123-D5/2E", 9074: "H-F123-D5/2W", 9086: "H-F1-D1-PAN", 9059: "H-F1-D5/2W-PAS",
                 9085: "H-F2-D5/2W-PAW", 9090: "H-PAW02",
                 9073: "H-PAW03", 9094: "H-PAW01", 9049: "W-ALL"})

    # Ahora vamos a obtener un resumen estadístico de las columnas de cada uno de los dos conjuntos de datos
    # obtenidos.

    # Antes debemos convertir las columnas del dataframe a flotante.
    sensor_values_group_1 = sensor_values_group_1.astype(np.float64)
    sensor_values_group_2 = sensor_values_group_2.astype(np.float64)

    print("Resumen estadístico de las columnas del primer conjunto de datos:\n")
    stats_group_1 = sensor_values_group_1.describe()
    print(stats_group_1)

    print("Resumen estadístico de las columnas del segundo conjunto de datos:\n")
    stats_group_2 = sensor_values_group_2.describe()
    print(stats_group_2)

    # Visualización de variables.

    # Fechas que queremos que aparezcan en las gráficas de las variables de cada conjunto de datos.
    dates_group_1 = [sensor_values_group_1.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                     sensor_values_group_1.index[2592].strftime("%Y-%m-%d %H:%M:%S"),
                     sensor_values_group_1.index[4992].strftime("%Y-%m-%d %H:%M:%S"),
                     sensor_values_group_1.index[7968].strftime("%Y-%m-%d %H:%M:%S")]
    dates_group_2 = [sensor_values_group_2.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                     sensor_values_group_2.index[2784].strftime("%Y-%m-%d %H:%M:%S"),
                     sensor_values_group_2.index[5760].strftime("%Y-%m-%d %H:%M:%S")]

    # Gráficas de los sensores que miden el consumo eléctrico.

    # Agrupamos los sensores eléctricos según su planta.
    electric_F1 = ["E-F1-D1", "E-F1-D5/2-C03", "E-F1-D5/2-C05", "E-F1-D5/2-C06", "E-F1-D5/2-C08"]
    electric_F2 = ["E-F2-D1", "E-F2-D2", "E-F2-D5/2-C11", "E-F2-D5/2-C12", "E-F2-D5/2-C15"]
    electric_F3 = ["E-F3-D1", "E-F3-D2", "E-F3-D5/2"]

    # Pintamos los valores de los sensores de la planta 1 en ambos conjuntos.
    src.data_analysis.plot_dataframe_variables(sensor_values_group_1[electric_F1],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_1)

    src.data_analysis.plot_dataframe_variables(sensor_values_group_2[electric_F1],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_2)

    # Pintamos los valores de los sensores de la planta 2 en ambos conjuntos.
    src.data_analysis.plot_dataframe_variables(sensor_values_group_1[electric_F2],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_1)

    src.data_analysis.plot_dataframe_variables(sensor_values_group_2[electric_F2],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_2)

    # Pintamos los valores de los sensores de la planta 3 en ambos conjuntos.
    src.data_analysis.plot_dataframe_variables(sensor_values_group_1[electric_F3],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_1)

    src.data_analysis.plot_dataframe_variables(sensor_values_group_2[electric_F3],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_2)

    # Gráficas de los sensores que miden el consumo de calefacción.

    # Agrupamos los sensores de calefacción según su planta.
    heating_123 = ["H-F123-D1", "H-F123-D5/2E", "H-F123-D5/2W"]
    heating_12 = ["H-F1-D1-PAN", "H-F1-D5/2W-PAS", "H-F2-D5/2W-PAW"]
    heating_PAW = ["H-PAW01", "H-PAW02", "H-PAW03"]

    # Pintamos los valores de los sensores que miden en las plantas 1, 2 y 3 en ambos conjuntos.
    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_1[heating_123],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_1)

    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_2[heating_123],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_2)

    # Pintamos los valores de los sensores que miden en las plantas 1 y 2 en ambos conjuntos.
    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_1[heating_12],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_1)

    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_2[heating_12],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_2)

    # Pintamos los valores de los sensores que miden en la zona piloto oeste.
    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_1[heating_PAW],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_1)

    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_2[heating_PAW],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="KW",
                                               xticks=dates_group_2)

    # Gráficas de los sensores de agua, temperatura y ocupación.
    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_1[["W-ALL", "Occupation", "Ext-Temp"]],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="",
                                               xticks=dates_group_1)

    src.data_analysis.plot_dataframe_variables(dataframe=sensor_values_group_2[["W-ALL", "Occupation"]],
                                               title="",
                                               xlabel="Tiempo",
                                               ylabel="",
                                               xticks=dates_group_2)

    # Preprocesamiento de los datos.

    # Transformación de datos.

    # Solo nos quedamos con el primer conjunto de datos, ya que los sensores de calefacción no registran consumo en septiembre, octubre y noviembre,

    # En primer lugar eliminamos la columna del sensor 9092, pues este tiene fallos en su funcionamiento.
    sensor_values_group_1 = sensor_values_group_1.drop([9092], axis=1)

    # Al eliminar la anterior columna, la cual contiene los valores de consumo de calefacción en las tres plantas  en
    # el área D2, también debemos eliminar todas las variables del área D2 pues no vamos a predecir el consumo de
    # calefacción en dicha área.
    sensor_values_group_1 = sensor_values_group_1.drop(["E-F2-D2", "E-F3-D2"], axis=1)

    # Eliminamos los sensores H-PAW01 y E-F1-D5/2-C08, pues no nos aportan ninguna información.
    sensor_values_group_1 = sensor_values_group_1.drop(["H-PAW01", "E-F1-D5/2-C08"], axis=1)

    # Ahora sumamos los valores de las columnas H-F123-D5/2W y H-F123-D5/2E para dar lugar a la columna
    # H-F123-D5/2.
    west_consumption = sensor_values_group_1["H-F123-D5/2W"].values
    east_consumption = sensor_values_group_1["H-F123-D5/2E"].values

    total_consumption = west_consumption + east_consumption

    sensor_values_group_1["H-F123-D5/2"] = total_consumption
    sensor_values_group_1 = sensor_values_group_1.drop(["H-F123-D5/2W", "H-F123-D5/2E"], axis=1)

    # Transformamos los datos para trabajar con el consumo real.
    # Antes es necesario eliminar las columnas Occupation y Ext-Temp y volverlas a añadir después.
    data_consumption_part_1 = src.data_analysis.get_difference(
        sensor_values_group_1.drop(["Occupation", "Ext-Temp"], axis=1).loc[
        "2016-01-05 00:00:00": "2016-02-25 23:45:00"])
    data_consumption_part_2 = src.data_analysis.get_difference(
        sensor_values_group_1.drop(["Occupation", "Ext-Temp"], axis=1).loc[
        "2016-03-01 00:00:00": "2016-04-25 23:45:00"])

    missing_values = pd.DataFrame(np.nan, index=pd.date_range(start="2016-02-26 00:00:00", end="2016-03-01 00:00:00",
                                                              freq="15min"),
                                  columns=data_consumption_part_1.columns, dtype=np.float64)

    # Unimos los dos conjuntos.
    data_consumption = pd.concat([data_consumption_part_1, missing_values, data_consumption_part_2], axis=0)

    # Sustituimos por 0 los valores de consumo negativos.
    data_consumption[data_consumption < 0] = 0.0

    # Añadimos la ocupación y la temperatura.
    data_consumption = src.data_analysis.add_occupation(data_consumption, weekly_occupation)
    data_consumption["Ext-Temp"] = np.concatenate(
        [temp_data.loc["2017-01-04 22:15:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-27 22:00:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-28 22:00:00":"2017-04-25 21:45:00", "SC000312"].values])

    # Descartamos los datos del mes de abril, puesto que no tenemos datos de consumo de calefacción.
    data_consumption = data_consumption.loc["2016-01-05 00:00:00":"2016-03-31 23:45:00"]

    # Tener en cuenta los días de puente para la ocupación del edificio ICPE.
    data_consumption["Occupation"].loc["2016-01-24 00:00:00":"2016-01-24 23:45:00"] = 0.0
    data_consumption["Occupation"].loc["2016-02-24 00:00:00":"2016-02-24 23:45:00"] = 0.0

    # Almacenamos el conjunto de datos en un csv para que pueda ser analizado y tratado en R.
    data_consumption.to_csv("../doc/datasets/data_consumption.csv", index_label="Datetime")


    # Coger el conjunto de datos generado en R sin outliers.
    data_consumption = pd.read_csv("../doc/datasets/data_consumption_no_outliers.csv", sep=";", index_col="Datetime",
                                   parse_dates=True)

    data_consumption = src.data_analysis.add_occupation(data_consumption, weekly_occupation)
    data_consumption["Occupation"].loc["2016-01-24 00:00:00":"2016-01-24 23:45:00"] = 0.0
    data_consumption["Occupation"].loc["2016-02-24 00:00:00":"2016-02-24 23:45:00"] = 0.0

    data_consumption["Ext-Temp"] = np.concatenate(
        [temp_data.loc["2017-01-04 22:15:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-27 22:00:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-28 22:00:00":"2017-03-31 21:45:00", "SC000312"].values])

    # Imputación de los valores perdidos.
    data_consumption_missing = data_consumption.copy(deep=True)

    # Imputación mediante MICE.
    data_consumption_imputed = data_consumption_missing.copy(deep=True)
    mice_imputer = IterativeImputer()
    data_consumption_imputed.iloc[:, :] = mice_imputer.fit_transform(data_consumption_missing)

    # Almacenar en un csv el conjunto de datos imputado para poder visualizarlo en R.
    data_consumption_imputed.to_csv("../doc/datasets/data_consumption_mice_imputed.csv")

    # Imputación de valores perdidos mediante patrones similares.

    # Eliminamos las columna de ocupación y temperatura, pues no tienen valore perdidos.
    data_consumption_missing = data_consumption_missing.drop(["Occupation", "Ext-Temp"], axis=1)
    sensor_values_group_1 = sensor_values_group_1.drop(["Occupation", "Ext-Temp"], axis=1)

    # Obtenemos el consumo total registrado por cada sensor los días 26, 27, 28 y 29.
    missing_days_consumption = np.reshape(np.diff(
        sensor_values_group_1.loc[[pd.Timestamp("2016-02-25 23:45:00"), pd.Timestamp("2016-03-01 00:00:00")]].values,
        axis=0), (18,))
    similar_past_week_days_consumption = np.reshape(np.sum(
        data_consumption_missing.loc[pd.Timestamp("2016-02-19 00:00:00"):pd.Timestamp("2016-02-23 00:00:00")].values,
        axis=0), (18,))
    similar_next_week_days_consumption = np.reshape(np.sum(
        data_consumption_missing.loc[pd.Timestamp("2016-03-04 00:00:00"):pd.Timestamp("2016-03-08 00:00:00")].values,
        axis=0), (18,))

    abs_diff_past_week = np.abs(np.sum(missing_days_consumption) - np.sum(similar_past_week_days_consumption))
    abs_diff_next_week = np.abs(np.sum(missing_days_consumption) - np.sum(similar_next_week_days_consumption))

    # Hacemos que los consumos de los dias perdidos sean iguales a los de los mismos días de la semana pasada.
    data_consumption_missing.loc["2016-02-26 00:00:00":"2016-03-01 00:00:00"] = data_consumption_missing.loc[
                                                                                "2016-02-19 00:00:00":"2016-02-23 00:00:00"].values

    # Añadimos la ocupación y la temperatura.
    data_consumption = src.data_analysis.add_occupation(data_consumption_missing, weekly_occupation)
    data_consumption["Occupation"].loc["2016-01-24 00:00:00":"2016-01-24 23:45:00"] = 0.0
    data_consumption["Occupation"].loc["2016-02-24 00:00:00":"2016-02-24 23:45:00"] = 0.0

    data_consumption["Ext-Temp"] = np.concatenate(
        [temp_data.loc["2017-01-04 22:15:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-27 22:00:00":"2017-02-28 21:45:00", "SC000312"].values,
         temp_data.loc["2017-02-28 22:00:00":"2017-03-31 21:45:00", "SC000312"].values])

    # Almacenar el conjunto de datos imputado en un fichero csv para que sea visualizado en R.
    data_consumption.to_csv("../doc/datasets/data_consumption_own_imputed.csv")

    # Selección de variables.

    # Calculamos la matriz de correlación.
    correlation_matrix = src.data_analysis.cross_correlation(data_consumption)

    # Pintamos un mapa de calor con la matriz de correlacion.
    src.data_analysis.plot_heatmap(correlation_matrix)

    # Obtener la importancia de las variables a la hora de predecir cada variable objetivo mediante xgb.

    # Variable H-F123-D1.

    # Gráfica de barras para representar visualmente la correlación de cada variable con la variable H-F123-D1.
    fig = plt.figure(dpi=300)
    plt.style.use("seaborn-darkgrid")
    plt.bar(range(20), np.abs(correlation_matrix.loc["H-F123-D1"].values))
    plt.title("Correlación con la variable H-F123-D1")
    plt.xlabel("Variable")
    plt.ylabel("Correlación")
    plt.xticks(range(20), correlation_matrix.columns, rotation=30)
    plt.show()

    # Determinamos la importancia de las variables predictoras mediante XGBoost.

    # Separar las variables predictoras y la variable objetivo.
    train_y = data_consumption["H-F123-D1"].values
    train_x = data_consumption.drop(["H-F123-D1"], axis=1)

    xgb_params = {
        "eta": 0.05,
        "max_depth": 6,
        "n_estimators": 10000,
        "subsample": 1.0,
        "colsample_bytree": 0.7,
        "objetive": "reg:linear",
        "eval_metric": "rmse",
        "silent": 4
    }

    # Entrenar el modelo.
    dtrain = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

    # Visualizar las variables por orden de importancia para predecir H-F123-D1.
    fig = plt.figure(dpi=300)
    plt.style.use("seaborn-darkgrid")
    xgb.plot_importance(model, max_num_features=19, height=0.8, ax=plt.gca())
    plt.show()

    # Variable H-F123-D5/2.

    # Gráfica de barras para representar visualmente la correlación de cada variable con la variable H-F123-D5/2.
    fig = plt.figure(dpi=300)
    plt.style.use("seaborn-darkgrid")
    plt.bar(range(20), np.abs(correlation_matrix.loc["H-F123-D5/2"].values))
    plt.title("Correlación con la variable H-F123-D5/2")
    plt.xlabel("Variable")
    plt.ylabel("Correlación")
    plt.xticks(range(20), correlation_matrix.columns, rotation=30)
    plt.show()

    # Determinamos la importancia de las variables predictoras mediante XGBoost.

    # Separar las variables predictoras y la variable objetivo.
    train_y = data_consumption["H-F123-D5/2"].values
    train_x = data_consumption.drop(["H-F123-D5/2"], axis=1)

    xgb_params = {
        "eta": 0.05,
        "max_depth": 6,
        "n_estimators": 10000,
        "subsample": 1.0,
        "colsample_bytree": 0.7,
        "objetive": "reg:linear",
        "eval_metric": "rmse",
        "silent": 4
    }

    # Entrenar el modelo.
    dtrain = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

    # Visualizar las variables por orden de importancia para predecir H-F123-D1.
    fig = plt.figure(dpi=300)
    plt.style.use("seaborn-darkgrid")
    xgb.plot_importance(model, max_num_features=19, height=0.8, ax=plt.gca())
    plt.show()

    # Modelos de predicción.

    # En primer lugar obtenemos el conjunto de datos con el que vamos a predecir cada variable objetivo.
    data_consumption_H_F123_D1 = data_consumption[
        ["H-F123-D1", "H-F123-D5/2", "Ext-Temp", "H-F1-D1-PAN", "Occupation", "H-PAW02", "H-PAW03"]]
    data_consumption_H_F123_D52 = data_consumption[
        ["H-F123-D5/2", "H-F123-D1", "Ext-Temp", "H-F1-D1-PAN", "Occupation", "H-PAW03", "E-F1-D5/2-C03"]]

    # Almacenamos en formato pkl el conjunto de datos con el que vamos a predecir H-F123-D1 para luego poder utilizarlo
    # en el script de experimentos.
    f = open("../doc/datasets/H_F123_D1_consumption.pkl", "wb")
    pickle.dump(data_consumption_H_F123_D1, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # Entrenamiento y validación con datos continuos.

    # Problema de enero.

    # Mejor modelo de XGBoost.

    # Modelo para predecir H_F123_D1.
    xgb_H_F123_D1 = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D1,
                                                  target="H-F123-D1",
                                                  lookback=384,
                                                  delay=48,
                                                  date_train_ini="2016-01-05 00:15:00",
                                                  date_train_end="2016-01-21 23:45:00",
                                                  date_val_ini="2016-01-22 00:00:00",
                                                  date_val_end="2016-01-31 23:45:00",
                                                  n_estimators=50,
                                                  max_depth=51,
                                                  learning_rate=0.05)

    # Entrenar el modelo.
    xgb_H_F123_D1.train()

    # Evaluamos el modelo.
    print(xgb_H_F123_D1.evaluateModel())

    xgb_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H_F123_D5/2.
    xgb_H_F123_D52 = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D52,
                                                   target="H-F123-D5/2",
                                                   lookback=384,
                                                   delay=48,
                                                   date_train_ini="2016-01-05 00:15:00",
                                                   date_train_end="2016-01-21 23:45:00",
                                                   date_val_ini="2016-01-22 00:00:00",
                                                   date_val_end="2016-01-31 23:45:00",
                                                   n_estimators=50,
                                                   max_depth=51,
                                                   learning_rate=0.05)

    # Entrenar el modelo.
    xgb_H_F123_D52.train()

    # Evaluamos el modelo.
    print(xgb_H_F123_D52.evaluateModel())

    xgb_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo MLP.

    # Modelo para predecir H-F123-D1.
    mlp_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-01-05 00:15:00",
                                                            date_train_end="2016-01-21 23:45:00",
                                                            date_val_ini="2016-01-22 00:00:00",
                                                            date_val_end="2016-01-31 23:45:00",
                                                            hidden_layers=[{"type": "Flatten"},
                                                                           {"type": "Dropout", "rate": 0.3},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"},
                                                                           {"type": "Dropout", "rate": 0.3},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"}],
                                                            optimizer="rmsprop",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Enero/mlp_H_F123_D1")

    # Entrenamos el modelo
    mlp_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=76,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Enero/mlp_H_F123_D1.hdf5")

    mlp_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    mlp_H_F123_D1.loadWeigths("../doc/models/Enero/mlp_H_F123_D1.hdf5")

    mlp_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    mlp_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-01-05 00:15:00",
                                                             date_train_end="2016-01-21 23:45:00",
                                                             date_val_ini="2016-01-22 00:00:00",
                                                             date_val_end="2016-01-31 23:45:00",
                                                             hidden_layers=[{"type": "Flatten"},
                                                                            {"type": "Dropout", "rate": 0.3},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"},
                                                                            {"type": "Dropout", "rate": 0.3},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"}],
                                                             optimizer="rmsprop",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Enero/mlp_H_F123_D52")

    # Entrenamos el modelo.
    mlp_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=76,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Enero/mlp_H_F123_D52.hdf5")

    mlp_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    mlp_H_F123_D52.loadWeigths("../doc/models/Enero/mlp_H_F123_D52.hdf5")

    mlp_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo RNN.

    # Modelo para predecir H-F123-D1.
    rnn_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-01-05 00:15:00",
                                                            date_train_end="2016-01-21 23:45:00",
                                                            date_val_ini="2016-01-22 00:00:00",
                                                            date_val_end="2016-01-31 23:45:00",
                                                            hidden_layers=[
                                                                {"type": "LSTM", "units": 32, "dropout": 0.15,
                                                                 "recurrent_dropout": 0.15,
                                                                 "return_sequences": True},
                                                                {"type": "LSTM", "units": 32,
                                                                 "dropout": 0.15,
                                                                 "recurrent_dropout": 0.15,
                                                                 "return_sequences": False},
                                                                {"type": "Dropout", "rate": 0.3},
                                                                {"type": "Dense", "units": 128,
                                                                 "activation": "relu"},
                                                                {"type": "Dropout", "rate": 0.3},
                                                                {"type": "Dense", "units": 128,
                                                                 "activation": "relu"}],
                                                            optimizer="rmsprop",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Enero/rnn_H_F123_D1")

    # Entrenamos el modelo
    rnn_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=76,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Enero/rnn_H_F123_D1.hdf5")

    rnn_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    rnn_H_F123_D1.loadWeigths("../doc/models/Enero/rnn_H_F123_D1.hdf5")

    rnn_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    rnn_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-01-05 00:15:00",
                                                             date_train_end="2016-01-21 23:45:00",
                                                             date_val_ini="2016-01-22 00:00:00",
                                                             date_val_end="2016-01-31 23:45:00",
                                                             hidden_layers=[
                                                                 {"type": "LSTM", "units": 32, "dropout": 0.15,
                                                                  "recurrent_dropout": 0.15,
                                                                  "return_sequences": True},
                                                                 {"type": "LSTM", "units": 32,
                                                                  "dropout": 0.15,
                                                                  "recurrent_dropout": 0.15,
                                                                  "return_sequences": False},
                                                                 {"type": "Dropout", "rate": 0.3},
                                                                 {"type": "Dense", "units": 128,
                                                                  "activation": "relu"},
                                                                 {"type": "Dropout", "rate": 0.3},
                                                                 {"type": "Dense", "units": 128,
                                                                  "activation": "relu"}],
                                                             optimizer="rmsprop",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Enero/rnn_H_F123_D52_prueba")

    # Entrenamos el modelo.
    rnn_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=76,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Enero/rnn_H_F123_D52_prueba.hdf5")

    rnn_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    rnn_H_F123_D52.loadWeigths("../doc/models/Enero/rnn_H_F123_D52_prueba.hdf5")

    rnn_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo CNN.

    # Modelo para predecir H-F123-D1.
    cnn_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-01-05 00:15:00",
                                                            date_train_end="2016-01-21 23:45:00",
                                                            date_val_ini="2016-01-22 00:00:00",
                                                            date_val_end="2016-01-31 23:45:00",
                                                            hidden_layers=[
                                                                {"type": "Conv1D", "filters": 32, "kernel_size": 5,
                                                                 "activation": "relu"},
                                                                {"type": "MaxPooling1D", "pool_size": 3},
                                                                {"type": "Conv1D", "filters": 32, "kernel_size": 5,
                                                                 "activation": "relu"},
                                                                {"type": "LSTM", "units": 32, "dropout": 0.15,
                                                                 "recurrent_dropout": 0.15,
                                                                 "return_sequences": True},
                                                                {"type": "LSTM", "units": 32,
                                                                 "dropout": 0.15,
                                                                 "recurrent_dropout": 0.15,
                                                                 "return_sequences": False},
                                                                {"type": "Dropout", "rate": 0.3},
                                                                {"type": "Dense", "units": 128,
                                                                 "activation": "relu"},
                                                                {"type": "Dropout", "rate": 0.3},
                                                                {"type": "Dense", "units": 128,
                                                                 "activation": "relu"}],
                                                            optimizer="rmsprop",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Enero/cnn_H_F123_D1")

    # Entrenamos el modelo
    cnn_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=76,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Enero/cnn_H_F123_D1.hdf5")

    cnn_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    cnn_H_F123_D1.loadWeigths("../doc/models/Enero/cnn_H_F123_D1.hdf5")

    cnn_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    cnn_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-01-05 00:15:00",
                                                             date_train_end="2016-01-21 23:45:00",
                                                             date_val_ini="2016-01-22 00:00:00",
                                                             date_val_end="2016-01-31 23:45:00",
                                                             hidden_layers=[
                                                                 {"type": "Conv1D", "filters": 32, "kernel_size": 5,
                                                                  "activation": "relu"},
                                                                 {"type": "MaxPooling1D", "pool_size": 3},
                                                                 {"type": "Conv1D", "filters": 32, "kernel_size": 5,
                                                                  "activation": "relu"},
                                                                 {"type": "LSTM", "units": 32, "dropout": 0.15,
                                                                  "recurrent_dropout": 0.15,
                                                                  "return_sequences": True},
                                                                 {"type": "LSTM", "units": 32,
                                                                  "dropout": 0.15,
                                                                  "recurrent_dropout": 0.15,
                                                                  "return_sequences": False},
                                                                 {"type": "Dropout", "rate": 0.3},
                                                                 {"type": "Dense", "units": 128,
                                                                  "activation": "relu"},
                                                                 {"type": "Dropout", "rate": 0.3},
                                                                 {"type": "Dense", "units": 128,
                                                                  "activation": "relu"}],
                                                             optimizer="rmsprop",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Enero/cnn_H_F123_D52")

    # Entrenamos el modelo.
    cnn_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=76,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Enero/cnn_H_F123_D52.hdf5")

    cnn_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    cnn_H_F123_D52.loadWeigths("../doc/models/Enero/cnn_H_F123_D52.hdf5")

    cnn_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo SeqToSeq.

    # Modelo para predecir H-F123-D1.
    seq2seq_H_F123_D1 = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D1,
                                                         target="H-F123-D1",
                                                         encoder_variables=["H-F123-D1", "H-F123-D5/2", "H-F1-D1-PAN",
                                                                            "H-PAW02", "H-PAW03"],
                                                         decoder_variables=["Occupation", "Ext-Temp"],
                                                         lookback=384,
                                                         delay=48,
                                                         hidden_layers=[64, 64],
                                                         optimizer="rmsprop",
                                                         loss="mae",
                                                         metrics=["mape"],
                                                         bidirectional=False,
                                                         logdir="../logs/BestModels/Enero/seq2seq_H_F123_D1",
                                                         date_train_ini="2016-01-05 00:15:00",
                                                         date_train_end="2016-01-21 23:45:00",
                                                         date_val_ini="2016-01-22 00:00:00",
                                                         date_val_end="2016-01-31 23:45:00"
                                                         )

    # Entrenamos el modelo.
    seq2seq_H_F123_D1.train(batch_train_size=72,
                            train_shuffle=True,
                            batch_val_size=76,
                            val_shuffle=False,
                            epochs=100,
                            checkpoint=True,
                            checkpoint_path="../doc/models/Enero/seq2seq_H_F123_D1.hdf5")

    seq2seq_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    seq2seq_H_F123_D1.loadWeigths("../doc/models/Enero/seq2seq_H_F123_D1.hdf5")

    seq2seq_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    seq2seq_H_F123_D52 = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D52,
                                                          target="H-F123-D5/2",
                                                          encoder_variables=["H-F123-D5/2", "H-F123-D1", "H-F1-D1-PAN",
                                                                             "H-PAW03", "E-F1-D5/2-C03"],
                                                          decoder_variables=["Occupation", "Ext-Temp"],
                                                          lookback=384,
                                                          delay=48,
                                                          hidden_layers=[64, 64],
                                                          optimizer="rmsprop",
                                                          loss="mae",
                                                          metrics=["mape"],
                                                          bidirectional=False,
                                                          logdir="../logs/BestModels/Enero/seq2seq_H_F123_D52",
                                                          date_train_ini="2016-01-05 00:15:00",
                                                          date_train_end="2016-01-21 23:45:00",
                                                          date_val_ini="2016-01-22 00:00:00",
                                                          date_val_end="2016-01-31 23:45:00"
                                                          )

    # Entrenamos el modelo.
    seq2seq_H_F123_D52.train(batch_train_size=72,
                             train_shuffle=True,
                             batch_val_size=76,
                             val_shuffle=False,
                             epochs=100,
                             checkpoint=True,
                             checkpoint_path="../doc/models/Enero/seq2seq_H_F123_D52.hdf5")

    seq2seq_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    seq2seq_H_F123_D52.loadWeigths("../doc/models/Enero/seq2seq_H_F123_D52.hdf5")

    seq2seq_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Problema de febrero.

    # Mejor modelo de XGBoost.

    # Modelo para predecir H_F123_D1.
    xgb_H_F123_D1 = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D1,
                                                  target="H-F123-D1",
                                                  lookback=384,
                                                  delay=48,
                                                  date_train_ini="2016-02-01 00:00:00",
                                                  date_train_end="2016-02-21 23:45:00",
                                                  date_val_ini="2016-02-22 00:00:00",
                                                  date_val_end="2016-02-29 23:45:00",
                                                  n_estimators=100,
                                                  max_depth=100,
                                                  learning_rate=0.001)

    # Entrenar el modelo.
    xgb_H_F123_D1.train()

    # Evaluamos el modelo.
    print(xgb_H_F123_D1.evaluateModel())

    xgb_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H_F123_D5/2.
    xgb_H_F123_D52 = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D52,
                                                   target="H-F123-D5/2",
                                                   lookback=384,
                                                   delay=48,
                                                   date_train_ini="2016-02-01 00:00:00",
                                                   date_train_end="2016-02-21 23:45:00",
                                                   date_val_ini="2016-02-22 00:00:00",
                                                   date_val_end="2016-02-29 23:45:00",
                                                   n_estimators=100,
                                                   max_depth=100,
                                                   learning_rate=0.001)

    # Entrenar el modelo.
    xgb_H_F123_D52.train()

    # Evaluamos el modelo.
    print(xgb_H_F123_D52.evaluateModel())

    xgb_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo MLP.

    # Modelo para predecir H-F123-D1.
    mlp_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-02-01 00:00:00",
                                                            date_train_end="2016-02-21 23:45:00",
                                                            date_val_ini="2016-02-22 00:00:00",
                                                            date_val_end="2016-02-29 23:45:00",
                                                            hidden_layers=[{"type": "Flatten"},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"}],
                                                            optimizer="adam",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Febrero/mlp_H_F123_D1")

    # Entrenamos el modelo
    mlp_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=64,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Febrero/mlp_H_F123_D1.hdf5")

    mlp_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    mlp_H_F123_D1.loadWeigths("../doc/models/Febrero/mlp_H_F123_D1.hdf5")

    mlp_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    mlp_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-02-01 00:00:00",
                                                             date_train_end="2016-02-21 23:45:00",
                                                             date_val_ini="2016-02-22 00:00:00",
                                                             date_val_end="2016-02-29 23:45:00",
                                                             hidden_layers=[{"type": "Flatten"},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"}],
                                                             optimizer="adam",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Febrero/mlp_H_F123_D52")

    # Entrenamos el modelo.
    mlp_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=64,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Febrero/mlp_H_F123_D52.hdf5")

    mlp_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    mlp_H_F123_D52.loadWeigths("../doc/models/Febrero/mlp_H_F123_D52.hdf5")

    mlp_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo RNN.

    # Modelo para predecir H-F123-D1.
    rnn_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-02-01 00:00:00",
                                                            date_train_end="2016-02-21 23:45:00",
                                                            date_val_ini="2016-02-22 00:00:00",
                                                            date_val_end="2016-02-29 23:45:00",
                                                            hidden_layers=[{"type": "LSTM", "units": 128,
                                                                            "dropout": 0.30,
                                                                            "recurrent_dropout": 0.30,
                                                                            "return_sequences": False},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"}],
                                                            optimizer="adam",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Febrero/rnn_H_F123_D1")

    # Entrenamos el modelo
    rnn_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=64,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Febrero/rnn_H_F123_D1.hdf5")

    rnn_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    rnn_H_F123_D1.loadWeigths("../doc/models/Febrero/rnn_H_F123_D1.hdf5")

    rnn_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    rnn_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-02-01 00:00:00",
                                                             date_train_end="2016-02-21 23:45:00",
                                                             date_val_ini="2016-02-22 00:00:00",
                                                             date_val_end="2016-02-29 23:45:00",
                                                             hidden_layers=[{"type": "LSTM", "units": 128,
                                                                             "dropout": 0.30,
                                                                             "recurrent_dropout": 0.30,
                                                                             "return_sequences": False},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"}],
                                                             optimizer="adam",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Febrero/rnn_H_F123_D52")

    # Entrenamos el modelo.
    rnn_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=64,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Febrero/rnn_H_F123_D52.hdf5")

    rnn_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    rnn_H_F123_D52.loadWeigths("../doc/models/Febrero/rnn_H_F123_D52.hdf5")

    rnn_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo CNN.

    # Modelo para predecir H-F123-D1.
    cnn_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-02-01 00:00:00",
                                                            date_train_end="2016-02-21 23:45:00",
                                                            date_val_ini="2016-02-22 00:00:00",
                                                            date_val_end="2016-02-29 23:45:00",
                                                            hidden_layers=[{"type": "Conv1D",
                                                                            "filters": 16,
                                                                            "kernel_size": 7,
                                                                            "activation": "relu"},
                                                                           {"type": "LSTM", "units": 128,
                                                                            "dropout": 0.30,
                                                                            "recurrent_dropout": 0.30,
                                                                            "return_sequences": False},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"}],
                                                            optimizer="adam",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Febrero/cnn_H_F123_D1")

    # Entrenamos el modelo
    cnn_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=64,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Febrero/cnn_H_F123_D1.hdf5")

    cnn_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    cnn_H_F123_D1.loadWeigths("../doc/models/Febrero/cnn_H_F123_D1.hdf5")

    cnn_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    cnn_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-02-01 00:00:00",
                                                             date_train_end="2016-02-21 23:45:00",
                                                             date_val_ini="2016-02-22 00:00:00",
                                                             date_val_end="2016-02-29 23:45:00",
                                                             hidden_layers=[
                                                                 {"type": "Conv1D",
                                                                  "filters": 16,
                                                                  "kernel_size": 7,
                                                                  "activation": "relu"},
                                                                 {"type": "LSTM", "units": 128,
                                                                  "dropout": 0.30,
                                                                  "recurrent_dropout": 0.30,
                                                                  "return_sequences": False},
                                                                 {"type": "Dropout", "rate": 0.15},
                                                                 {"type": "Dense", "units": 128,
                                                                  "activation": "relu"}],
                                                             optimizer="adam",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Febrero/cnn_H_F123_D52")

    # Entrenamos el modelo.
    cnn_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=64,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Febrero/cnn_H_F123_D52.hdf5")

    cnn_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    cnn_H_F123_D52.loadWeigths("../doc/models/Febrero/cnn_H_F123_D52.hdf5")

    cnn_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo SeqToSeq.

    # Modelo para predecir H-F123-D1.
    seq2seq_H_F123_D1 = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D1,
                                                         target="H-F123-D1",
                                                         lookback=384,
                                                         delay=48,
                                                         encoder_variables=["H-F123-D1", "H-F123-D5/2",
                                                                            "H-F1-D1-PAN", "H-PAW02", "H-PAW03"],
                                                         decoder_variables=["Occupation", "Ext-Temp"],
                                                         date_train_ini="2016-02-01 00:00:00",
                                                         date_train_end="2016-02-21 23:45:00",
                                                         date_val_ini="2016-02-22 00:00:00",
                                                         date_val_end="2016-02-29 23:45:00",
                                                         hidden_layers=[64, 64],
                                                         optimizer="adam",
                                                         loss="mae",
                                                         metrics=["mape"],
                                                         bidirectional=False,
                                                         logdir="../logs/BestModels/Febrero/seq2seq_H_F123_D1")

    # Entrenamos el modelo
    seq2seq_H_F123_D1.train(batch_train_size=72,
                            train_shuffle=True,
                            batch_val_size=64,
                            val_shuffle=False,
                            epochs=100,
                            checkpoint=True,
                            checkpoint_path="../doc/models/Febrero/seq2seq_H_F123_D1.hdf5")

    seq2seq_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    seq2seq_H_F123_D1.loadWeigths("../doc/models/Febrero/seq2seq_H_F123_D1.hdf5")

    seq2seq_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    seq2seq_H_F123_D52 = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D52,
                                                          target="H-F123-D5/2",
                                                          lookback=384,
                                                          delay=48,
                                                          encoder_variables=["H-F123-D5/2", "H-F123-D1",
                                                                             "H-F1-D1-PAN", "H-PAW03",
                                                                             "E-F1-D5/2-C03"],
                                                          decoder_variables=["Occupation", "Ext-Temp"],
                                                          date_train_ini="2016-02-01 00:00:00",
                                                          date_train_end="2016-02-21 23:45:00",
                                                          date_val_ini="2016-02-22 00:00:00",
                                                          date_val_end="2016-02-29 23:45:00",
                                                          hidden_layers=[64, 64],
                                                          optimizer="adam",
                                                          loss="mae",
                                                          metrics=["mape"],
                                                          bidirectional=False,
                                                          logdir="../logs/BestModels/Febrero/seq2seq_H_F123_D52")

    # Entrenamos el modelo.
    seq2seq_H_F123_D52.train(batch_train_size=72,
                             train_shuffle=True,
                             batch_val_size=64,
                             val_shuffle=False,
                             epochs=100,
                             checkpoint=True,
                             checkpoint_path="../doc/models/Febrero/seq2seq_H_F123_D52.hdf5")

    seq2seq_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    seq2seq_H_F123_D52.loadWeigths("../doc/models/Febrero/seq2seq_H_F123_D52.hdf5")

    seq2seq_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Problema de marzo.

    # Mejor modelo de XGBoost.

    # Modelo para predecir H_F123_D1.
    xgb_H_F123_D1 = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D1,
                                                  target="H-F123-D1",
                                                  lookback=384,
                                                  delay=48,
                                                  date_train_ini="2016-03-01 00:00:00",
                                                  date_train_end="2016-03-21 23:45:00",
                                                  date_val_ini="2016-03-22 00:00:00",
                                                  date_val_end="2016-03-31 23:45:00",
                                                  n_estimators=50,
                                                  max_depth=51,
                                                  learning_rate=0.05)

    # Entrenar el modelo.
    xgb_H_F123_D1.train()

    # Evaluamos el modelo.
    print(xgb_H_F123_D1.evaluateModel())

    xgb_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H_F123_D5/2.
    xgb_H_F123_D52 = src.XGBoost.XGBoostTimeSeries(dataframe=data_consumption_H_F123_D52,
                                                   target="H-F123-D5/2",
                                                   lookback=384,
                                                   delay=48,
                                                   date_train_ini="2016-03-01 00:00:00",
                                                   date_train_end="2016-03-21 23:45:00",
                                                   date_val_ini="2016-03-22 00:00:00",
                                                   date_val_end="2016-03-31 23:45:00",
                                                   n_estimators=50,
                                                   max_depth=51,
                                                   learning_rate=0.05)

    # Entrenar el modelo.
    xgb_H_F123_D52.train()

    # Evaluamos el modelo.
    print(xgb_H_F123_D52.evaluateModel())

    xgb_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo MLP.

    # Modelo para predecir H-F123-D1.
    mlp_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-03-01 00:00:00",
                                                            date_train_end="2016-03-21 23:45:00",
                                                            date_val_ini="2016-03-22 00:00:00",
                                                            date_val_end="2016-03-31 23:45:00",
                                                            hidden_layers=[{"type": "Flatten"},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"}],
                                                            optimizer="rmsprop",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Marzo/mlp_H_F123_D1")

    # Entrenamos el modelo
    mlp_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=76,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Marzo/mlp_H_F123_D1.hdf5")

    mlp_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    mlp_H_F123_D1.loadWeigths("../doc/models/Marzo/mlp_H_F123_D1.hdf5")

    mlp_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    mlp_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-03-01 00:00:00",
                                                             date_train_end="2016-03-21 23:45:00",
                                                             date_val_ini="2016-03-22 00:00:00",
                                                             date_val_end="2016-03-31 23:45:00",
                                                             hidden_layers=[{"type": "Flatten"},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"}],
                                                             optimizer="rmsprop",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Marzo/mlp_H_F123_D52")

    # Entrenamos el modelo.
    mlp_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=76,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Marzo/mlp_H_F123_D52.hdf5")

    mlp_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    mlp_H_F123_D52.loadWeigths("../doc/models/Marzo/mlp_H_F123_D52.hdf5")

    mlp_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo RNN.

    # Modelo para predecir H-F123-D1.
    rnn_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-03-01 00:00:00",
                                                            date_train_end="2016-03-21 23:45:00",
                                                            date_val_ini="2016-03-22 00:00:00",
                                                            date_val_end="2016-03-31 23:45:00",
                                                            hidden_layers=[{"type": "LSTM", "units": 128,
                                                                            "dropout": 0.50,
                                                                            "recurrent_dropout": 0.50,
                                                                            "return_sequences": False},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"}],
                                                            optimizer="rmsprop",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Marzo/rnn_H_F123_D1")

    # Entrenamos el modelo
    rnn_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=76,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Marzo/rnn_H_F123_D1.hdf5")

    rnn_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    rnn_H_F123_D1.loadWeigths("../doc/models/Marzo/rnn_H_F123_D1.hdf5")

    rnn_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    rnn_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-03-01 00:00:00",
                                                             date_train_end="2016-03-21 23:45:00",
                                                             date_val_ini="2016-03-22 00:00:00",
                                                             date_val_end="2016-03-31 23:45:00",
                                                             hidden_layers=[{"type": "LSTM", "units": 128,
                                                                             "dropout": 0.50,
                                                                             "recurrent_dropout": 0.50,
                                                                             "return_sequences": False},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"}],
                                                             optimizer="rmsprop",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Marzo/rnn_H_F123_D52")

    # Entrenamos el modelo.
    rnn_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=76,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Marzo/rnn_H_F123_D52.hdf5")

    rnn_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    rnn_H_F123_D52.loadWeigths("../doc/models/Marzo/rnn_H_F123_D52.hdf5")

    rnn_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo CNN.

    # Modelo para predecir H-F123-D1.
    cnn_H_F123_D1 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D1,
                                                            target="H-F123-D1",
                                                            lookback=384,
                                                            delay=48,
                                                            date_train_ini="2016-03-01 00:00:00",
                                                            date_train_end="2016-03-21 23:45:00",
                                                            date_val_ini="2016-03-22 00:00:00",
                                                            date_val_end="2016-03-31 23:45:00",
                                                            hidden_layers=[{"type": "Conv1D",
                                                                            "filters": 16,
                                                                            "kernel_size": 5,
                                                                            "activation": "relu"},
                                                                           {"type": "LSTM", "units": 128,
                                                                            "dropout": 0.50,
                                                                            "recurrent_dropout": 0.50,
                                                                            "return_sequences": False},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"},
                                                                           {"type": "Dropout", "rate": 0.15},
                                                                           {"type": "Dense", "units": 128,
                                                                            "activation": "relu"}],
                                                            optimizer="rmsprop",
                                                            loss="mae",
                                                            metrics=["mape"],
                                                            logdir="../logs/BestModels/Marzo/cnn_H_F123_D1")

    # Entrenamos el modelo
    cnn_H_F123_D1.train(batch_train_size=72,
                        train_shuffle=True,
                        batch_val_size=76,
                        val_shuffle=False,
                        epochs=100,
                        checkpoint=True,
                        checkpoint_path="../doc/models/Marzo/cnn_H_F123_D1.hdf5")

    cnn_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    cnn_H_F123_D1.loadWeigths("../doc/models/Marzo/cnn_H_F123_D1.hdf5")

    cnn_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    cnn_H_F123_D52 = src.NeuronalNetworks.BasicTimeSeriesANN(dataframe=data_consumption_H_F123_D52,
                                                             target="H-F123-D5/2",
                                                             lookback=384,
                                                             delay=48,
                                                             date_train_ini="2016-03-01 00:00:00",
                                                             date_train_end="2016-03-21 23:45:00",
                                                             date_val_ini="2016-03-22 00:00:00",
                                                             date_val_end="2016-03-31 23:45:00",
                                                             hidden_layers=[{"type": "Conv1D",
                                                                             "filters": 16,
                                                                             "kernel_size": 5,
                                                                             "activation": "relu"},
                                                                            {"type": "LSTM", "units": 128,
                                                                             "dropout": 0.50,
                                                                             "recurrent_dropout": 0.50,
                                                                             "return_sequences": False},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"},
                                                                            {"type": "Dropout", "rate": 0.15},
                                                                            {"type": "Dense", "units": 128,
                                                                             "activation": "relu"}],
                                                             optimizer="rmsprop",
                                                             loss="mae",
                                                             metrics=["mape"],
                                                             logdir="../logs/BestModels/Marzo/cnn_H_F123_D52")

    # Entrenamos el modelo.
    cnn_H_F123_D52.train(batch_train_size=72,
                         train_shuffle=True,
                         batch_val_size=76,
                         val_shuffle=False,
                         epochs=100,
                         checkpoint=True,
                         checkpoint_path="../doc/models/Marzo/cnn_H_F123_D52.hdf5")

    cnn_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    cnn_H_F123_D52.loadWeigths("../doc/models/Marzo/cnn_H_F123_D52.hdf5")

    cnn_H_F123_D52.plot_prediction(title="Predicción valores de validación")

    # Mejor modelo SeqToSeq.

    # Modelo para predecir H-F123-D1.
    seq2seq_H_F123_D1 = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D1,
                                                         target="H-F123-D1",
                                                         lookback=384,
                                                         delay=48,
                                                         encoder_variables=["H-F123-D1", "H-F123-D5/2",
                                                                            "H-F1-D1-PAN", "H-PAW02", "H-PAW03"],
                                                         decoder_variables=["Occupation", "Ext-Temp"],
                                                         date_train_ini="2016-03-01 00:00:00",
                                                         date_train_end="2016-03-21 23:45:00",
                                                         date_val_ini="2016-03-22 00:00:00",
                                                         date_val_end="2016-03-31 23:45:00",
                                                         hidden_layers=[64, 64],
                                                         optimizer="rmsprop",
                                                         loss="mae",
                                                         metrics=["mape"],
                                                         bidirectional=False,
                                                         logdir="../logs/BestModels/Marzo/seq2seq_H_F123_D1")

    # Entrenamos el modelo
    seq2seq_H_F123_D1.train(batch_train_size=72,
                            train_shuffle=True,
                            batch_val_size=76,
                            val_shuffle=False,
                            epochs=100,
                            checkpoint=True,
                            checkpoint_path="../doc/models/Marzo/seq2seq_H_F123_D1.hdf5")

    seq2seq_H_F123_D1.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    seq2seq_H_F123_D1.loadWeigths("../doc/models/Marzo/seq2seq_H_F123_D1.hdf5")

    seq2seq_H_F123_D1.plot_prediction(title="Predicción valores de validación")

    # Modelo para predecir H-F123-D5/2.
    seq2seq_H_F123_D52 = src.NeuronalNetworks.SeqToSeqANN(dataframe=data_consumption_H_F123_D52,
                                                          target="H-F123-D5/2",
                                                          lookback=384,
                                                          delay=48,
                                                          encoder_variables=["H-F123-D5/2", "H-F123-D1",
                                                                             "H-F1-D1-PAN", "H-PAW03",
                                                                             "E-F1-D5/2-C03"],
                                                          decoder_variables=["Occupation", "Ext-Temp"],
                                                          date_train_ini="2016-03-01 00:00:00",
                                                          date_train_end="2016-03-21 23:45:00",
                                                          date_val_ini="2016-03-22 00:00:00",
                                                          date_val_end="2016-03-31 23:45:00",
                                                          hidden_layers=[64, 64],
                                                          optimizer="rmsprop",
                                                          loss="mae",
                                                          metrics=["mape"],
                                                          bidirectional=False,
                                                          logdir="../logs/BestModels/Marzo/seq2seq_H_F123_D52")

    # Entrenamos el modelo.
    seq2seq_H_F123_D52.train(batch_train_size=72,
                             train_shuffle=True,
                             batch_val_size=76,
                             val_shuffle=False,
                             epochs=100,
                             checkpoint=True,
                             checkpoint_path="../doc/models/Marzo/seq2seq_H_F123_D52.hdf5")

    seq2seq_H_F123_D52.plotHistory(metric="loss", title="NMAE de entrenamiento y validación")

    seq2seq_H_F123_D52.loadWeigths("../doc/models/Marzo/seq2seq_H_F123_D52.hdf5")

    seq2seq_H_F123_D52.plot_prediction(title="Predicción valores de validación")
