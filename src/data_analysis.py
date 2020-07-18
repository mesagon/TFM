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
# - data_analysis.py: Script que contiene un conjunto de funciones empleadas para llevar a cabo distintas tareas de
#  análisis exploratorio y preprocesamiento.
######################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Función para añadir a un conjunto de datos con valores de sensores del edificio ICPE una columna
# con el porcentaje de ocupación para cada instante de tiempo (timestep).
#
#   - Parámetros:
#
#       - dataframe: Pandas dataframe con el conjunto de datos al que queremos añadir la columna.
#       - weekly_occupation: Pandas dataframe con la distribución de ocupación semanal del edificio ICPE.
#
#   - Valor devuelto:
#
#       - df: Pandas dataframe recibido como argumento con la columna de ocupación.
def add_occupation(dataframe, weekly_occupation):
    # Lista en la que vamos almacenando los valores de la columna ocupación.
    occupation = []

    # Rango de fechas diarias entre el primer y el último timestep del conjunto de datos.
    daily_dates = pd.date_range(start=dataframe.index[0].strftime("%Y-%m-%d"),
                                end=dataframe.index[-1].strftime("%Y-%m-%d"), freq="1d")

    # Recorremos los días.
    for d in daily_dates:

        # Obtenemos los timesteps del día actual.
        act_day_df = dataframe.loc[d.strftime("%Y-%m-%d")]

        # Puede que en el día actual no haya medidas de sensores.
        if (not act_day_df.empty):
            # Extraemos las horas del día actual.
            act_day_hours = act_day_df.index.strftime("%H:%M:%S").to_list()

            # Obtenemos en una lista los valores de ocupación para cada una de las horas del día actual.
            act_day_occ = weekly_occupation.loc[act_day_hours].iloc[:, d.dayofweek].values

            # Añadimos los valores de ocupación del día actual a la lista.
            occupation.extend(act_day_occ)

    df = dataframe.copy(deep=True)

    df["Occupation"] = occupation

    return (df)


# Función que recibe como argumento un conjunto de datos y pinta en una misma figura una gráfica de líneas por
# cada variable del mismo. Cada gráfica mostrará la evolución de los valores de la variable a lo largo del tiempo.
# Las variables del conjunto de datos deben de estar expresadas en la misma unidad.
#
#   - Parámetros:
#
#       - dataframe: Pandas Dataframe con el conjunto de datos cuyas variables deseamos pintar.
#       - title: Título del conjunto de gráficas.
#       - xlabel: Etiqueta del eje x de todas las gráficas.
#       - ylabel: Etiqueta del eje y de todas las gráficas.
#       - xticks: Lista con los valores del eje x que queremos que aparezcan en las gráficas.
def plot_dataframe_variables(dataframe, title, xlabel, ylabel, xticks):
    # Inicializamos la figura.
    fig = plt.figure(dpi=400)
    plt.style.use("seaborn-darkgrid")

    # Creamos la paleta de colores.
    palette = plt.get_cmap("Set1")

    # Pintamos una gráfica por cada variable del dataframe.
    num = 0

    for colum in dataframe.columns:

        num += 1

        plt.subplot(dataframe.values.shape[-1], 1, num)

        # Pintar los valores de la variable actual en un gráfico de lineas.
        plt.plot(dataframe.index.strftime("%Y-%m-%d %H:%M:%S"), dataframe[colum], marker='', color=palette(num),
                 linewidth=1.0, alpha=0.9, label=colum)

        # En el eje x no pintamos todas las fechas.
        plt.xticks(xticks)
        plt.ylabel(ylabel)

        # No pintamos ticks en todas las gráficas.
        if (num != dataframe.values.shape[-1]):

            plt.tick_params(labelbottom="off")

        else:

            plt.xlabel(xlabel)

        plt.title(colum, fontsize=12, fontweight=0, color='black')

    # Título general.
    plt.suptitle(title, y=0.95, fontsize=13, fontweight=0, color='black', style='italic')


# Función que pinta un mapa de calor a partir de una matriz de correlación.
#
#   - Parámetros:
#
#       - dataframe: Pandas dataframe con la matriz de correlación.
def plot_heatmap(dataframe):
    # Generamos una máscara, mediante la cual indicaremos que solo queremos utlizar los valores de la matriz de correlación
    # que se encuentren debajo de la diagonal para pintar el mapa de calor.
    mask = np.triu(np.ones_like(dataframe, dtype=np.bool))

    # Creamos la figura en la que vamos a pintar el mapa de calor.
    fig, ax = plt.subplots(figsize=(11, 9))

    # Creamos una paleta de colores.
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Pintamos finalmente el mapa de calor.
    sns.heatmap(dataframe, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Función que recibe como argumento un pandas dataframe y devuelve otro con las mismas columnas pero que contiene la
# diferencia entre las filas del primero.
#
#   - Parámetros:
#
#       - dataframe: Pandas Dataframe de cuyas filas se obtendrá la diferencia.
#
#   - Valor devuelto:
#
#       - diff_dataframe: Pandas Dataframe que contiene la diferencia entre las filas del Pandas Dataframe recibido
#       como argumento.
def get_difference(dataframe):
    # Pandas dataframe en el que vamos a almacenar la diferencia.
    diff_dataframe = pd.DataFrame(columns=dataframe.columns,
                                  index=pd.date_range(start=dataframe.index[1].strftime("%Y-%m-%d %H:%M:%S"),
                                                      end=dataframe.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                                                      freq="15min"))

    # Obtenemos la diferencia entre las filas del dataframe en un numpy array.
    diff_values = np.diff(dataframe.values, axis=0)

    diff_dataframe.iloc[:, :] = diff_values

    return (diff_dataframe)


# Función que recibe un conjunto de datos y devuelve una matriz con la correlación cruzada entre cada par de columnas
# del mismo. La correlación cruzada entre cada par de variables se calcula desplazando n (siendo n el número de
# observaciones) veces un timestep una de ellas, obteniendo en cada desplazamiento el coeficiente de correlación de
# pearson entre ambas variables. Finalmente, de todos los valores obtenidos se toma el mayor como el índice de correlación.
#
#   - Parámetros:
#
#       - dataframe: Pandas dataframe que contiene las variables entre las que vamos a obtener la correlación cruzada.
#
#   - Valor devuelto:
#
#       - correlation_matrix: Matriz que en la casilla i,j contendrá la correlación cruzada entre la variable i y
#       la variable j.
def cross_correlation(dataframe):
    # Matriz en la que almacenaremos las correlaciones.
    correlation_matrix = pd.DataFrame(index=dataframe.columns, columns=dataframe.columns, dtype=np.float64)

    # Recorremos las columnas del conjunto de datos.
    for i, c in enumerate(dataframe.columns):

        # Obtenemos la correlación de la columna actual consigo misma.
        max_corr = np.max(np.correlate(dataframe.loc[:, c].values, dataframe.loc[:, c].values, "same"))

        # Obtenemos la correlación de cada columna con la columna actual.
        for j, cc in enumerate(dataframe.columns):

            # Normalizamos los dos vectores.
            a = dataframe[c].values
            b = dataframe[cc].values

            a = (a - np.mean(a)) / np.std(a)
            b = (b - np.mean(b)) / np.std(b)

            # Obtenemos la correlación entre ambas columnas.
            max_corr = np.max(np.correlate(a, b, "full")) / len(a)
            min_corr = np.min(np.correlate(a, b, "full")) / len(a)

            if (np.abs(max_corr) > np.abs(min_corr)):

                correlation_matrix.loc[c, cc] = max_corr

            else:

                correlation_matrix.loc[c, cc] = min_corr

        print(i)

    return (correlation_matrix)