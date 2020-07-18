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
# - preprocessing.R: Script que contiene el código necesario para llevar a cabo las tareas de preprocesamiento de
#   visualización de los datos transformados, tratamiento de outliers y tratamiento de valores perdidos.
######################################################################################################################

library(tidyverse)
library(TSA)
library(ggplot2)
library(ggthemes)
library(zoo)
library(forecast)
library(Amelia)
library(mtsdi)
library(VIM)
library(mice)
library(missForest)
library(gridExtra)

# Leemos el conjunto de datos
data_raw <- read_csv("../doc/datasets/data_consumption.csv")

# Funciones.
graphs = 0
decomposition <- function(df,sensors){
  
  # Recorremos los sensores.
  for (s in sensors){
    
    series <- ts(pull(df,s), frequency = 672)
    series[is.na(series)] <- 0.0
    decomp <- stl(series, s.window = "periodic")[["time.series"]]
    
    # Pintamos la descomposición con ggplot.
    
    tb <- tibble(Datetime = df$`Datetime`, Data = as.numeric(series), 
                 Seasonal = as.numeric(decomp[,1]), Trend = as.numeric(decomp[,2]),
                 Remainder = as.numeric(decomp[,3]))
    
    
    g1 = ggplot(data = tb, aes(x = Datetime, y = Data)) + labs(x="Tiempo", y = "Datos") + theme_grey()+ geom_line(color="#F08080")
    g2 = ggplot(data = tb, aes(x = Datetime, y = Seasonal)) + labs(x="Tiempo", y = "Estacionalidad") + theme_grey()+ geom_line(color="#F08080")
    g3 = ggplot(data = tb, aes(x = Datetime, y = Trend)) + labs(x="Tiempo", y = "Tendencia") + theme_grey()+ geom_line(color="#F08080") 
    g4 = ggplot(data = tb, aes(x = Datetime, y = Remainder)) + labs(x="Tiempo", y = "Irregularidad") + theme_grey()+ geom_line(color="#F08080")
    
    # Pintamos todas las gráficas a la vez.                             
    grid.arrange(g1,g2,g3,g4,nrow=4,ncol=1, top = s )
  }
}

autocorrelation <- function(df,sensors){

  # Recorremos los sensores. 
  for (s in sensors){
    
    series <- ts(pull(df,s), frequency = 672)
    series[is.na(series)] <- 0.0
    bacf <- acf(series, plot = FALSE)
    bacfdf <- with(bacf, data.frame(lag,acf))
    
    g <- ggplot(data = bacfdf, mapping = aes(x = lag, y = acf)) +
        geom_hline(aes(yintercept = 0)) +
        geom_segment(mapping = aes(xend = lag, yend = 0)) + 
        ggtitle(paste("                                                                        ", s))
   

    print(g)
  
  }
}

# Revisualización de las variables.

# En primer lugar transformamos los valores de todas las columnas a flotante.
data = type_convert(data_raw)

# Sensores eléctricos.

  # Planta 1.
  
  # Descomposición.
  decomposition(data,c("E-F1-D1", "E-F1-D5/2-C03", "E-F1-D5/2-C05","E-F1-D5/2-C06"))

  # Autocorrelación.
  autocorrelation(data,c("E-F1-D1", "E-F1-D5/2-C03", "E-F1-D5/2-C05","E-F1-D5/2-C06"))
  
  # Planta 2.
  
  # Descomposición.
  decomposition(data,c("E-F2-D1","E-F2-D5/2-C11","E-F2-D5/2-C12","E-F2-D5/2-C15"))
  
  # Autocorrelación.
  autocorrelation(data,c("E-F2-D1","E-F2-D5/2-C11","E-F2-D5/2-C12","E-F2-D5/2-C15"))
  
  # Planta 3.
  
  # Descomposición
  decomposition(data,c("E-F3-D1","E-F3-D5/2"))
  
  # Autocorrelación.
  autocorrelation(data,c("E-F3-D1","E-F3-D5/2"))
  
# Sensores de calefacción.
  
  # Plantas 1, 2 y 3.
  
  # Descomposición.
  decomposition(data,c("H-F123-D1", "H-F123-D5/2"))
  
  # Autocorrelación.
  autocorrelation(data,c("H-F123-D1", "H-F123-D5/2"))
  
  # Plantas 1 y 2.
  
  # Descomposición.
  decomposition(data,c("H-F1-D5/2W-PAS", "H-F1-D1-PAN","H-PAW02", "H-PAW03", "H-F2-D5/2W-PAW"))
  
  # Autocorrelación.
  autocorrelation(data,c("H-F1-D5/2W-PAS", "H-F1-D1-PAN","H-PAW02", "H-PAW03", "H-F2-D5/2W-PAW"))
  
# Sensores de agua, ocupación, y temperatura exterior.
  
  # Descomposición.
  decomposition(data,c("W-ALL", "Occupation", "Ext-Temp"))
  
  # Autocorrelación.
  autocorrelation(data,c("W-ALL", "Occupation", "Ext-Temp"))
  
# Tratamiento de outliers.

par(mfrow = c(2,1))

  # Variable E-F1-D1.
  series <- ts(data$`E-F1-D1`, frequency = 672)
  ts.plot(series)

  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)
  data$`E-F1-D1` = as.numeric(series)
  
  # Variable E-F1-D5/2-C03.
  series <- ts(data$`E-F1-D5/2-C03`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)
  data$`E-F1-D5/2-C03` = as.numeric(series)
  
  # Variable E-F1-D5/2-C05.
  series <- ts(data$`E-F1-D5/2-C05`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)
  data$`E-F1-D5/2-C05` = as.numeric(series)
  
  # Variable E-F1-D5/2-C06.
  series <- ts(data$`E-F1-D5/2-C06`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)
  data$`E-F1-D5/2-C06` = as.numeric(series)
  
  # Variable E-F2-D1.
  series <- ts(data$`E-F2-D1`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)
  data$`E-F2-D1` = as.numeric(series)

  # Variable E-F2-D5/2-C11.
  series <- ts(data$`E-F2-D5/2-C11`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)
  data$`E-F2-D5/2-C11` = as.numeric(series)

  # Variable E-F2-D5/2-C12.
  series <- ts(data$`E-F2-D5/2-C12`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  
  # Variable E-F2-D5/2-C15.
  series <- ts(data$`E-F2-D5/2-C15`, frequency = 672)
  ts.plot(series)
  
  g1 <- ggplot(data = data, aes(x = Datetime, y = `E-F2-D5/2-C15` )) +
    labs(x="Tiempo", y = "Consumo (KW)") + 
    theme_grey() +
    geom_line(color="#F08080") +
    ggtitle("                                                            E-F2-D5/2-C15 con outliers")
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  
  data$`E-F2-D5/2-C15` = as.numeric(series)
  
  g2 <- ggplot(data = data, aes(x = Datetime, y = `E-F2-D5/2-C15` )) +
    labs(x="Tiempo", y = "Consumo (KW)") + 
    theme_grey() +
    geom_line(color="#F08080") +
    ggtitle("                                                            E-F2-D5/2-C15 sin outliers")
  
  grid.arrange(g1,g2,nrow=2,ncol=1)
  
  # Variable E-F3-D1.
  series <- ts(data$`E-F3-D1`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  
  # Variable E-F3-D5/2.
  series <- ts(data$`E-F3-D5/2`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)   
  data$`E-F3-D5/2` = as.numeric(series)
  
  # Variable H-F123-D1.
  series <- ts(data$`H-F123-D1`, frequency = 672)
  
  g1 <- ggplot(data = data, aes(x = Datetime, y = `H-F123-D1` )) +
    labs(x="Tiempo", y = "Consumo") + 
    theme_grey() +
    geom_line(color="#F08080") +
    ggtitle("                                                             H-F123-D1 con outliers")
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>12] = 12
  
  data$`H-F123-D1` = as.numeric(series)
  
  g2 <- ggplot(data = data, aes(x = Datetime, y = `H-F123-D1` )) +
    labs(x="Tiempo", y = "Consumo") + 
    theme_grey() +
    geom_line(color="#F08080") +
    ggtitle("                                                             H-F123-D1 sin outliers")
  
  
  grid.arrange(g1,g2,nrow=2,ncol=1)
  
  # Variable H-F123-D5/2.
  series <- ts(data$`H-F123-D5/2`, frequency = 672)

  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>68] = 68

  data$`H-F123-D5/2` = as.numeric(series)
  
  # Variable H-F1-D5/2W-PAS.
  series <- ts(data$`H-F1-D5/2W-PAS`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series) 
  data$`H-F1-D5/2W-PAS` = as.numeric(series)
  
  # Variable H-F1-D1-PAN.
  series <- ts(data$`H-F1-D1-PAN`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>3] = 3
  ts.plot(series) 
  data$`H-F1-D1-PAN` = as.numeric(series)

  # Variable H-PAW02.
  series <- ts(data$`H-PAW02`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series)
  data$`H-PAW02` = as.numeric(series)
  
  # Variable H-PAW03.
  series <- ts(data$`H-PAW03`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>1] = 1
  ts.plot(series) 
  data$`H-PAW03` = as.numeric(series)
  
  # Variable H-F2-D5/2W-PAW.
  series <- ts(data$`H-F2-D5/2W-PAW`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>2] = 2
  ts.plot(series) 
  data$`H-F2-D5/2W-PAW` = as.numeric(series)
  
  # Variable W-ALL.
  series <- ts(data$`W-ALL`, frequency = 672)
  
  g1 <- ggplot(data = data, aes(x = Datetime, y = `W-ALL` )) +
    labs(x="Tiempo", y = "Consumo") + 
    theme_grey() +
    geom_line(color="#F08080") +
    ggtitle("                                                             W-ALL con outliers")
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  series[series>10] = 10
  
  data$`W-ALL` = as.numeric(series)
  
  g2 <- ggplot(data = data, aes(x = Datetime, y = `W-ALL` )) +
    labs(x="Tiempo", y = "Consumo") + 
    theme_grey() +
    geom_line(color="#F08080") +
    ggtitle("                                                             W-ALL sin outliers")
  
  grid.arrange(g1,g2,nrow=2,ncol=1)
  
  # Variable Occupation.
  series <- ts(data$Occupation, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  
  # Variable Ext-Temp.
  series <- ts(data$`Ext-Temp`, frequency = 672)
  ts.plot(series)
  
  quantile(series, na.rm = TRUE, prob=seq(0, 1, length = 101))
  
  # Guardamos el conjunto de datos sin outliers en un csv.
  write_csv2(data, "../doc/datasets/data_consumption_no_outliers.csv")
  
# Imputación de valores perdidos.
  
  # Técnicas de imputación multivariable.
  
    # Amelia.
    imputation <- Amelia::amelia(data, m = 1, ts = "Datetime", polytime = 2)
    imputed_data <- imputation[["imputations"]][[1]]
    
    tb <- tibble(`H-F123-D1` = sum(imputed_data$`H-F123-D1`[4992:5376]),
                 `E-F1-D5/2-C03` = sum(imputed_data$`E-F1-D5/2-C03`[4992:5376]),
                 `W-ALL` = sum(imputed_data$`W-ALL`[4992:5376]))
    
    g1 <- ggplot(data = imputed_data, aes(x = Datetime, y = `H-F123-D1` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#F08080") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black", size = 0.2) +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black", size = 0.2) +
      ggtitle("                                                             H-F123-D1 imputada")
    
    g11 <- ggplot(data = tb, aes(x = "H-F123-D1", y = `H-F123-D1` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#F08080") +
      geom_hline(yintercept = 1071, linetype = "dashed", color = "black")
    
    g2 <- ggplot(data = imputed_data, aes(x = Datetime, y = `E-F1-D5/2-C03` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#87CEFA") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             E-F1-D5/2-C03 imputada")
    
    g21 <- ggplot(data = tb, aes(x = "E-F1-D5/2-C03", y = `E-F1-D5/2-C03` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#87CEFA") +
      geom_hline(yintercept = 157, linetype = "dashed", color = "black")
    
    g3 <- ggplot(data = imputed_data, aes(x = Datetime, y = `W-ALL` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#3CB371") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             W-ALL imputada")
    
    g31 <- ggplot(data = tb, aes(x = "W-ALL", y = `W-ALL` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#3CB371") +
      geom_hline(yintercept = 1670, linetype = "dashed", color = "black")
    
    grid.arrange(g1,g11,g2,g21,g3,g31,nrow=3,ncol=2, widths = c(5,1))
    
    # Mtsdi.
    imputation <- mnimput(~`E-F2-D5/2-C12`+`E-F2-D1`+`E-F3-D1`+`E-F2-D5/2-C15`+`E-F3-D5/2`
                          +`E-F1-D5/2-C03`+`E-F1-D5/2-C05`+`E-F1-D1`+`E-F1-D5/2-C06`+`E-F2-D5/2-C11`
                          +`W-ALL`+`H-F1-D5/2W-PAS`+`H-PAW03`+`H-F2-D5/2W-PAW`+`H-F1-D1-PAN`
                          +`H-PAW02`+`H-F123-D1`+`H-F123-D5/2`+ Occupation + `Ext-Temp`,
                          data, ts = TRUE, method = "spline")
    
    imputed_data <- imputation[["filled.dataset"]]
    
    tb <- tibble(`H-F123-D1` = sum(imputed_data$`H-F123-D1`[4992:5376]),
                 `E-F1-D5/2-C03` = sum(imputed_data$`E-F1-D5/2-C03`[4992:5376]),
                 `W-ALL` = sum(imputed_data$`W-ALL`[4992:5376]))
    
    g1 <- ggplot(data = imputed_data, aes(x = data$Datetime, y = `H-F123-D1` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#F08080") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black", size = 0.2) +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black", size = 0.2) +
      ggtitle("                                                             H-F123-D1 imputada")
    
    g11 <- ggplot(data = tb, aes(x = "H-F123-D1", y = `H-F123-D1` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#F08080") +
      geom_hline(yintercept = 1071, linetype = "dashed", color = "black")
    
    g2 <- ggplot(data = imputed_data, aes(x = data$Datetime, y = `E-F1-D5/2-C03` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#87CEFA") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             E-F1-D5/2-C03 imputada")
    
    g21 <- ggplot(data = tb, aes(x = "E-F1-D5/2-C03", y = `E-F1-D5/2-C03` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#87CEFA") +
      geom_hline(yintercept = 157, linetype = "dashed", color = "black")
    
    g3 <- ggplot(data = imputed_data, aes(x = data$Datetime, y = `W-ALL` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#3CB371") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             W-ALL imputada")
    
    g31 <- ggplot(data = tb, aes(x = "W-ALL", y = `W-ALL` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#3CB371") +
      geom_hline(yintercept = 1670, linetype = "dashed", color = "black")
    
    grid.arrange(g1,g11,g2,g21,g3,g31,nrow=3,ncol=2, widths = c(5,1))
    
    # Irmi.
    imputed_data <-irmi(data)
    
    tb <- tibble(`H-F123-D1` = sum(imputed_data$`H-F123-D1`[4992:5376]),
                 `E-F1-D5/2-C03` = sum(imputed_data$`E-F1-D5/2-C03`[4992:5376]),
                 `W-ALL` = sum(imputed_data$`W-ALL`[4992:5376]))
    
    g1 <- ggplot(data = imputed_data, aes(x = Datetime, y = `H-F123-D1` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#F08080") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black", size = 0.2) +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black", size = 0.2) +
      ggtitle("                                                             H-F123-D1 imputada")
    
    g11 <- ggplot(data = tb, aes(x = "H-F123-D1", y = `H-F123-D1` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#F08080") +
      geom_hline(yintercept = 1071, linetype = "dashed", color = "black")
    
    g2 <- ggplot(data = imputed_data, aes(x = Datetime, y = `E-F1-D5/2-C03` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#87CEFA") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             E-F1-D5/2-C03 imputada")
    
    g21 <- ggplot(data = tb, aes(x = "E-F1-D5/2-C03", y = `E-F1-D5/2-C03` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#87CEFA") +
      geom_hline(yintercept = 157, linetype = "dashed", color = "black")
    
    g3 <- ggplot(data = imputed_data, aes(x = Datetime, y = `W-ALL` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#3CB371") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             W-ALL imputada")
    
    g31 <- ggplot(data = tb, aes(x = "W-ALL", y = `W-ALL` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#3CB371") +
      geom_hline(yintercept = 1670, linetype = "dashed", color = "black")
    
    grid.arrange(g1,g11,g2,g21,g3,g31,nrow=3,ncol=2, widths = c(5,1))
    
    # mice.
    # Leemos el conjunto de datos imputado en python.
    imputed_data <- read_csv("../doc/datasets/data_consumption_mice_imputed.csv")
    imputed_data = type_convert(imputed_data)
    
    tb <- tibble(`H-F123-D1` = sum(imputed_data$`H-F123-D1`[4992:5376]),
                 `E-F1-D5/2-C03` = sum(imputed_data$`E-F1-D5/2-C03`[4992:5376]),
                 `W-ALL` = sum(imputed_data$`W-ALL`[4992:5376]))
    
    g1 <- ggplot(data = imputed_data, aes(x = Datetime, y = `H-F123-D1` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#F08080") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black", size = 0.2) +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black", size = 0.2) +
      ggtitle("                                                             H-F123-D1 imputada")
    
    g11 <- ggplot(data = tb, aes(x = "H-F123-D1", y = `H-F123-D1` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#F08080") +
      geom_hline(yintercept = 1071, linetype = "dashed", color = "black")
    
    g2 <- ggplot(data = imputed_data, aes(x = Datetime, y = `E-F1-D5/2-C03` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#87CEFA") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             E-F1-D5/2-C03 imputada")
    
    g21 <- ggplot(data = tb, aes(x = "E-F1-D5/2-C03", y = `E-F1-D5/2-C03` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#87CEFA") +
      geom_hline(yintercept = 157, linetype = "dashed", color = "black")
    
    g3 <- ggplot(data = imputed_data, aes(x = Datetime, y = `W-ALL` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#3CB371") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             W-ALL imputada")
    
    g31 <- ggplot(data = tb, aes(x = "W-ALL", y = `W-ALL` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#3CB371") +
      geom_hline(yintercept = 1670, linetype = "dashed", color = "black")
    
    grid.arrange(g1,g11,g2,g21,g3,g31,nrow=3,ncol=2, widths = c(5,1))
    
  # Técnicas de imputación univariable.
    
    # StructTS.
    series <- ts(data$`H-F123-D1`, frequency = 20)
    H_F123_D1 <- as.numeric(na.StructTS(series))
    
    series <- ts(data$`E-F1-D5/2-C03`, frequency = 20)
    E_F1_D52_C03 <- as.numeric(na.StructTS(series))
    
    series <- ts(data$`W-ALL`, frequency = 20)
    W_ALL <- as.numeric(na.StructTS(series))
    
    imputed_data <- tibble(Datetime = data$Datetime, `H-F123-D1` = H_F123_D1, `E-F1-D5/2-C03` = E_F1_D52_C03, `W-ALL` = W_ALL)
    
    tb <- tibble(`H-F123-D1` = sum(imputed_data$`H-F123-D1`[4992:5376]),
                 `E-F1-D5/2-C03` = sum(imputed_data$`E-F1-D5/2-C03`[4992:5376]),
                 `W-ALL` = sum(imputed_data$`W-ALL`[4992:5376]))
    
    g1 <- ggplot(data = imputed_data, aes(x = Datetime, y = `H-F123-D1` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#F08080") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black", size = 0.2) +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black", size = 0.2) +
      ggtitle("                                                             H-F123-D1 imputada")
    
    g11 <- ggplot(data = tb, aes(x = "H-F123-D1", y = `H-F123-D1` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#F08080") +
      geom_hline(yintercept = 1071, linetype = "dashed", color = "black")
    
    g2 <- ggplot(data = imputed_data, aes(x = Datetime, y = `E-F1-D5/2-C03` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#87CEFA") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             E-F1-D5/2-C03 imputada")
    
    g21 <- ggplot(data = tb, aes(x = "E-F1-D5/2-C03", y = `E-F1-D5/2-C03` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#87CEFA") +
      geom_hline(yintercept = 157, linetype = "dashed", color = "black")
    
    g3 <- ggplot(data = imputed_data, aes(x = Datetime, y = `W-ALL` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#3CB371") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             W-ALL imputada")
    
    g31 <- ggplot(data = tb, aes(x = "W-ALL", y = `W-ALL` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#3CB371") +
      geom_hline(yintercept = 1670, linetype = "dashed", color = "black")
    
    grid.arrange(g1,g11,g2,g21,g3,g31,nrow=3,ncol=2, widths = c(5,1))
    
    # interp.
    series <- ts(data$`H-F123-D1`, frequency = 672)
    H_F123_D1 <- as.numeric(na.interp(series))
    
    series <- ts(data$`E-F1-D5/2-C03`, frequency = 672)
    E_F1_D52_C03 <- as.numeric(na.interp(series))
    
    series <- ts(data$`W-ALL`, frequency = 672)
    W_ALL <- as.numeric(na.interp(series))
    
    imputed_data <- tibble(Datetime = data$Datetime, `H-F123-D1` = H_F123_D1, `E-F1-D5/2-C03` = E_F1_D52_C03, `W-ALL` = W_ALL)
    
    tb <- tibble(`H-F123-D1` = sum(imputed_data$`H-F123-D1`[4992:5376]),
                 `E-F1-D5/2-C03` = sum(imputed_data$`E-F1-D5/2-C03`[4992:5376]),
                 `W-ALL` = sum(imputed_data$`W-ALL`[4992:5376]))
    
    g1 <- ggplot(data = imputed_data, aes(x = Datetime, y = `H-F123-D1` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#F08080") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black", size = 0.2) +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black", size = 0.2) +
      ggtitle("                                                             H-F123-D1 imputada")
    
    g11 <- ggplot(data = tb, aes(x = "H-F123-D1", y = `H-F123-D1` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#F08080") +
      geom_hline(yintercept = 1071, linetype = "dashed", color = "black")
    
    g2 <- ggplot(data = imputed_data, aes(x = Datetime, y = `E-F1-D5/2-C03` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#87CEFA") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             E-F1-D5/2-C03 imputada")
    
    g21 <- ggplot(data = tb, aes(x = "E-F1-D5/2-C03", y = `E-F1-D5/2-C03` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#87CEFA") +
      geom_hline(yintercept = 157, linetype = "dashed", color = "black")
    
    g3 <- ggplot(data = imputed_data, aes(x = Datetime, y = `W-ALL` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#3CB371") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             W-ALL imputada")
    
    g31 <- ggplot(data = tb, aes(x = "W-ALL", y = `W-ALL` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#3CB371") +
      geom_hline(yintercept = 1670, linetype = "dashed", color = "black")
    
    grid.arrange(g1,g11,g2,g21,g3,g31,nrow=3,ncol=2, widths = c(5,1))
    
  # Imputación por patrones similares.
    
    # Leer el conjunto de datos imputados en python.
    imputed_data <- read_csv("../doc/datasets/data_consumption_own_imputed.csv")
    imputed_data <- type_convert(imputed_data)
    
    tb <- tibble(`H-F123-D1` = sum(imputed_data$`H-F123-D1`[4992:5376]),
                 `E-F1-D5/2-C03` = sum(imputed_data$`E-F1-D5/2-C03`[4992:5376]),
                 `W-ALL` = sum(imputed_data$`W-ALL`[4992:5376]))
    
    g1 <- ggplot(data = imputed_data, aes(x = Datetime, y = `H-F123-D1` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#F08080") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black", size = 0.2) +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black", size = 0.2) +
      ggtitle("                                                             H-F123-D1 imputada")
    
    g11 <- ggplot(data = tb, aes(x = "H-F123-D1", y = `H-F123-D1` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#F08080") +
      geom_hline(yintercept = 1071, linetype = "dashed", color = "black")
    
    g2 <- ggplot(data = imputed_data, aes(x = Datetime, y = `E-F1-D5/2-C03` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#87CEFA") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             E-F1-D5/2-C03 imputada")
    
    g21 <- ggplot(data = tb, aes(x = "E-F1-D5/2-C03", y = `E-F1-D5/2-C03` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#87CEFA") +
      geom_hline(yintercept = 157, linetype = "dashed", color = "black")
    
    g3 <- ggplot(data = imputed_data, aes(x = Datetime, y = `W-ALL` )) +
      labs(x="Tiempo", y = "Consumo (KW)") + 
      theme_grey() +
      geom_line(color="#3CB371") +
      geom_vline(xintercept = data$Datetime[4950], linetype="dashed", color = "black") +
      geom_vline(xintercept = data$Datetime[5450], linetype="dashed", color = "black") +
      ggtitle("                                                             W-ALL imputada")
    
    g31 <- ggplot(data = tb, aes(x = "W-ALL", y = `W-ALL` )) +
      labs(x=" ", y = " ") + 
      theme_grey() +
      geom_bar(stat = "identity", width = 0.2, fill = "#3CB371") +
      geom_hline(yintercept = 1670, linetype = "dashed", color = "black")
    
    grid.arrange(g1,g11,g2,g21,g3,g31,nrow=3,ncol=2, widths = c(5,1))
  