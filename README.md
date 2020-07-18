# Deep learning para simulación energética de edificios

En los últimos años los edificios residenciales, de oficina e industriales han
supuesto más de un tercio del consumo energético mundial, razón por la cual
cada vez es mayor el interés en mejorar la eficiencia energética de los mismos.
En este trabajo de fin de méster se ha abordado un problema de creación
de modelos de predicción del consumo de energía en un edificio bajo las
estrategias de operación normales —esto es, sin modificar los parámetros de
control de los equipos de calefacción, aire acondicionado, iluminación, etc— a
partir de datos históricos y con alto nivel de precisión, de forma que dichos
modelos sirvan a los ingenieros de energía responsables del edificio para
caracterizar el funcionamiento base del mismo e identificar oportunidades
de mejora.

Específicamente, en este trabajo se ha aplicado conjunto de tareas de
análisis y preprocesamiento a un conjunto de datos real que contiene valores
históricos de consumo energético registrados a lo largo del año 2016 por los
sensores de un edificio de oficinas denominado ICPE, situado en Rumanía.
Todo esto con el objetivo de en última instancia poder aplicar a dicho conjunto
de datos una serie de técnicas de aprendizaje automático y en especial
de aprendizaje profundo que permitan predecir a partir de valores históricos
de varios sensores el consumo futuro de energía destinada a los sistemas de
calefacción del edificio.

Como consecuencia de las tareas de análisis y preprocesamiento solo se
ha podido trabajar con valores históricos de sensores de una zona piloto del
edificio correspondientes a los meses de Enero, Febrero y Marzo de 2016, lo
que acarrea, entre otros, el principal inconveniente de no disponer de una
gran variedad y cantidad de datos con los que entrenar los distintos modelos.
Sin embargo, a pesar de esto hemos podido comprobar que las técnicas de
aprendizaje profundo y en especial aquellas especializadas en el tratamiento
de series temporales nos ofrecen modelos con una gran capacidad de aprendizaje
y de generalización a la hora de resolver este problema de predicción
planteado.
