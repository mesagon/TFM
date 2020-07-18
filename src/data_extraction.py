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
# - data_extraction.py: Script que contiene la clase ICPEDAtaExtractor, la cual proporciona los atributos y métodos
#  necesarios para conectarse al API que nos permite acceder a los datos de los sensores del edificio ICPE y extraerlos.
######################################################################################################################

import requests
import pandas as pd
import numpy as np
import json

class ICPEDataExtractor:

    # Constructor de la clase.
    def __init__(self, username, password):

        # Inicializamos los atributos.
        self._url_login = "http://150.214.203.11:5000/auth/login" # Url para autenticarnos en el API.
        self._url_sensors_interval = "http://150.214.203.11:5000/processing/icpe/date/interval" # Url que nos permite obtener los datos de los sensores del edificio ICPE entre una fecha de inicio y una fecha final.
        self._url_sensors_day = "http://150.214.203.11:5000/processing/icpe/date" # Url que nos permite obtener los datos de los sensores del edificio ICPE de un dia concreto.

        try:

            # Abrimos una sesión con el API y la almacenamos en un atributo, de forma que pueda ser utilizada por los métodos de la clase
            # para realizar peticiones al API.
            self._session = requests.session()

            # Parámetros que pasamos a la url del API para logearnos.
            params = {"username": username, "password": password}

            # Nos autenticamos en el API.
            response = self._session.post(self._url_login, json = params)

        except requests.exceptions.RequestException as e:

            print(e)
            exit(1)

    # Destructor de la clase.
    def __del__(self):

        # Cerramos la sesión del API.
        self._session.close()

    # Método para obtener la url de login.
    def getUrlLogin(self):

        return(self._url_login)

    # Método para establecer la url de login.
    def setUrlLogin(self, url):

        self._url_login = url

    # Método para obtener la url de sensors interval.
    def getUrlSensorsInterval(self):

        return(self._url_sensors_interval)

    # Método para establecer la url de sensors interval.
    def setUrlSensorsInterval(self, url):

        self._url_sensors_interval = url

    # Método para obtener la url de sensors day.
    def getUrlSensorsDay(self):

        return(self._url_sensors_day)

    # Método para establecer la url de sensors day.
    def setUrlSensorsDay(self, url):

        self._url_sensors_day = url

    # Método para obtener la sesión.
    def getSession(self):

        return(self._session)

    # Método para obtener la sesión.
    def setSession(self,session):

        self._session = session


    # Método para obtener los sensores del edificio ICPE que realizan mediciones todos los días
    # entre una fecha de inicio y otra final.
    #
    #   Argumentos:
    #
    #       - date_init: String con la fecha de inicio en formato yyyy-mm-dd.
    #       - date_end: String con la fecha final en formato yyyy-mm-dd.
    #
    #   Valor devuelto:
    #
    #       - sensors: Conjunto con los IDs de los sensores que realizan mediciones todos los días
    #                   entre date_init y date_end.
    def everydaySensors(self, date_init, date_end):

        try:

            # En primer lugar pedimos al API todos los datos de los sensores del edificio icpe entre
            # date_ini y date_end.
            response = self._session.post(self._url_sensors_interval, json = {"date_init": date_init, "date_end": date_end})

            # Obtenemos la respuesta como un diccionario para acceder a los datos de los sensores.
            response_dict = json.loads(response.text)
            sensors_data = response_dict["result"]

            # Lista en la que vamos a almacenar por cada día entre date_init y date_end un conjunto
            # con los IDs de los sensores que realizan mediciones ese día.
            diary_sensors = []

            daily_dates = pd.date_range(start=date_init, end=date_end, freq="1D")

            for date in daily_dates:

                # Conjunto en el que almacenar los IDs de los sensores que realizan mediciones en
                # la fecha actual.
                sensors_actual_date = set()

                # Recorremos los sensores.
                for id_sensor in sensors_data:

                    # Si en la fecha actual el sensor actual ha realizado mediciones, ponemos
                    # su ID en el conjunto.
                    if (date.strftime("%Y-%m-%d") in sensors_data[id_sensor]):
                        sensors_actual_date.add(int(id_sensor))

                diary_sensors.append(sensors_actual_date)

            # Finalmente realizamos la intersección de todos los conjuntos de sensores diarios obtenidos. Con
            # lo que conseguiremos obtener los IDs de los sensores que realizan mediciones todos los días.
            sensors = set.intersection(*diary_sensors)

            return(sensors)

        except requests.exceptions.RequestException as e:

            print(e)
            exit(1)


    # Método para obtener en un pandas dataframe los valores captados por los sensores del edificio ICPE
    # entre una fecha de inicio y otra final. Solo estarán en el dataframe final los valores de aquellos sensores
    # que miden todos los días entre la fecha de inicio y la final.
    #
    #   Argumentos:
    #
    #       - date_init: String con la fecha de inicio en formato yyyy-mm-dd.
    #       - date_end: String con la fecha final en formato yyyy-mm-dd.
    #
    #   Valor devuelto:
    #
    #       - data: Dataframe con los valores registrados por los sensores entre date_init y date_end. En cada fila de este dataframe
    #               se encontrarán los valores registrados por cada sensor (columnas) en una fecha y hora concretas.
    #
    def sensorsValues(self, date_init, date_end):


        # En primer lugar obtenemos el conjunto de sensores que realizan mediciones entre date_init y date_end.
        sensors = self.everydaySensors(date_init, date_end)
        sensors = list(sensors) # Pasamos el conjunto de sensores a lista.
        sensors.sort() # Ordenamos los sensores.

        try:

            # Diccionario en el que almacenamos los datos de los sensores según fecha y hora.
            sensor_values_date_hour = {}

            minutely_dates = pd.date_range(start = date_init + " 00:00:00", end = date_end + " 23:45:00", freq="15min")

            # Inicializamos el diccionario en el que almacenaremos los valores de los sensores.
            for single_date in minutely_dates:

                sensor_values_date_hour[single_date.strftime("%Y-%m-%d %H:%M:%S")] = []

            i = 0

            daily_dates = pd.date_range(start = date_init, end = date_end, freq = "1D")

            # Recogemos los datos.
            for single_date in daily_dates:

                # Pedimos al API que nos de los datos de los sensores del día actual.
                response = self._session.post(self._url_sensors_day, json = {"day": single_date.strftime("%Y-%m-%d")})

                # Convertimos la respuesta a diccionario para poder procesarla.
                resp_dict = json.loads(response.text)
                sensors_data = resp_dict["result"]

                for data in sensors_data:

                    if(int(data["ID_Sensor"]) in sensors):

                        # Obtener fecha y hora de este valor de sensor.
                        data_date = data["day"] + " " + data["time"]
                        sensor_values_date_hour[data_date].append({"ID_Sensor": data["ID_Sensor"], "value": data["value"]})

                print(i)
                i += 1

            # Finalmente creamos la serie temporal.
            data = pd.DataFrame(data=np.zeros(shape=(len(sensor_values_date_hour), len(sensors))),
                                       index = minutely_dates, columns = sensors)

            i = 0

            for single_date in minutely_dates:

                # Obtenemos del diccionario la lista de valores registrados en la fecha y hora actual.
                single_date_values = sensor_values_date_hour[single_date.strftime("%Y-%m-%d %H:%M:%S")]

                for value in single_date_values:

                     data.loc[single_date.strftime("%Y-%m-%d %H:%M:%S"), value["ID_Sensor"]] = value["value"]

                print(i)
                i += 1

            return(data)

        except requests.exceptions.RequestException as e:

            print(e)
            exit(1)