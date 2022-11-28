from regressors import SimpleLinearRegressor, MultipleLinearRegressor
from classifiers import KNN
from distances import euclidean
import numpy as np
import pandas as pd
import regression_metrics as rm
import classification_metrics as cm
import sklearn.metrics

def prueba_1():
    # Obtenemos los valores X1, X2 e Y
    Y = np.array([113, 118, 127, 132, 136, 144, 138, 146, 156, 149])
    X1 = np.array([20, 20, 25, 25, 30, 30, 30, 40, 40, 40])
    X2 = np.array([1, 2, 1, 2, 1, 2, 3, 1, 2, 3])
    # Generamos los Regresores Lineales Simples
    linear_1 = SimpleLinearRegressor(X1, Y)
    linear_2 = SimpleLinearRegressor(X2, Y)
    linear_3 = MultipleLinearRegressor(np.row_stack((X1, X2)), Y)
    # Obtenemos para cada regresor simple sus parámetros beta_0 y beta_1
    b0_1, b1_1 = linear_1.get_parameters()
    b0_2, b1_2 = linear_2.get_parameters()
    b0_3, b1_3, b2_3 = linear_3.get_parameters()
    # Mostramos por pantalla la ecuación de cada regresor
    print(f'Regresor para X1 -> Y = {b0_1} + {b1_1}*X1')
    print(f'Regresor para X2 -> Y = {b0_2} + {b1_2}*X2')
    print(f'Regresor para X1 y X2 -> Y = {b0_3} + {b1_3}*X1 + {b2_3}*X2')
    # Cargamos el conjunto de test
    Y_test = np.array([200, 116, 122, 130, 150, 120, 146, 155, 156, 147])
    X1_test = np.array([35, 25, 25, 20, 35, 25, 42, 35, 40, 42])
    X2_test = np.array([1, 2, 2, 1, 2, 2, 1, 1, 2, 2]) 
    # Obtenemos las predicciones de cada regresor
    Y_pred_1 = linear_1.predict(X1_test)
    Y_pred_2 = linear_2.predict(X2_test)
    Y_pred_3 = linear_3.predict(np.row_stack((X1_test, X2_test)))
    # Obtenemos las métricas de cada regresor
    funcs = [rm.mean_absolute_error, rm.mean_squared_error, rm.root_mean_squared_error, rm.coef_det]
    Ys = [Y_pred_1, Y_pred_2, Y_pred_3]
    metrics = [list(map(func, [Y_test] * len(Ys), Ys)) for func in funcs]
    # Mostramos los resultados como un dataFrame
    met = ["M1", "M2", "M3"]
    res = pd.DataFrame({"MAE":metrics[0], "MSE":metrics[1], "RMSE":metrics[2], "R^2":metrics[3]}, index = met)
    print(f'Métricas de regresión -> \n {res}')


def prueba_2():
    # Definimos los vectores de los coeficientes beta
    beta_1 = np.array([8.1743, 21.8065, 4.5648, -26.3083, -43.887])
    beta_2 = np.array([42.637, 2.4652, 6.6809, -9.4293, -18.2859])
    # Definimos la matriz de características
    X = np.array([[4.6, 3.2, 1.4, 0.2],
                [5.3, 3.7, 1.5, 0.2],
                [5.7, 4.4, 1.5, 0.4],
                [5.0, 3.5, 1.6, 0.6],
                [5.5, 2.5, 4.0, 1.3],
                [5.7, 3.0, 4.2, 1.2],
                [5.7, 2.8, 4.2, 1.3],
                [5.8, 2.7, 5.1, 1.9],
                [6.3, 2.5, 5.0, 1.9],
                [5.9, 3.0, 5.1, 1.8]])
    # Definimos el vector de clases
    classes = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    # Calculamos f1 sobre todos los puntos
    f_1 = beta_1[0] + np.dot(beta_1[1:], np.transpose(X))
    # Calculamos f2 sobre todos los puntos
    f_2 = beta_2[0] + np.dot(beta_2[1:], np.transpose(X))
    # Calculamos la exponencial de dichas funciones
    exp_f1 = np.exp(f_1)
    exp_f2 = np.exp(f_2)
    # Calculamos la probabilidad de pertenencia a cada clase
    sum_exp = exp_f1 + exp_f2
    prob_y1 = exp_f1 / (1 + sum_exp)
    prob_y2 = exp_f2 / (1 + sum_exp)
    prob_y3 = 1 - prob_y1 - prob_y2
    list_prob = list(zip(prob_y1, prob_y2, prob_y3))
    classes_pred = np.array([tup.index(max(tup)) for tup in list_prob]) + 1
    # Mostramos los resultados obtenidos en un DataFrame
    res_df = pd.DataFrame({"exp(f1)":exp_f1, "exp(f2)":exp_f2, "sum(exp)":sum_exp,
                            "p(Y = 1|X)":prob_y1, "p(Y = 2|X)":prob_y2, "p(Y = 3|X)":prob_y3,
                            "Clase Predicha":classes_pred, "Clase Real":classes}, index = list(range(1,11)))
    print(res_df)
    # Calculamos las métricas de rendimiento
    confusion_matrix = cm.confusion_matrix(classes, classes_pred)
    funcs = [cm.accuracy, cm.recall, cm.fp_rate, cm.specificity, cm.precission, cm.f1_score, cm.kappa]
    metrics = [func(confusion_matrix) for func in funcs]
    res_df = pd.DataFrame({"CCR":metrics[0], "Recall":metrics[1], "Fp Rate":metrics[2],
                            "Especifidad":metrics[3], "Precision":metrics[4], "F1 Score":metrics[5],
                            "Kappa":metrics[6]}, index = list(range(1,4)))
    print(f"\n\nMetricas de clasificación -> \n{res_df}")

def prueba_3():
    X = np.array([[4.6, 3.2, 1.4],
                [5.3, 3.7, 1.5],
                [5.7, 4.4, 1.5],
                [5.0, 3.5, 1.6],
                [5.5, 2.5, 4.0],
                [5.7, 3.0, 4.2],
                [5.7, 2.8, 4.1],
                [5.8, 2.7, 5.1],
                [6.3, 2.5, 5.0],
                [5.9, 3.0, 5.1]])
    classes = np.array([1, 1, 1, 2, 1, 2, 2, 1, 2, 2])
    knn = KNN(X, classes, 3, euclidean)
    X_pred = np.array([[5, 3.5, 1.7],
                [4.3, 2.8, 1.5],
                [2.7, 4.5, 1.2],
                [5.0, 4.2, 1.3],
                [6.3, 2.5, 4.1],
                [5.2, 3.0, 4.5],
                [4.5, 3, 4.2],
                [5.9, 2.9, 5.1],
                [5, 2.4, 5.1],
                [4.5, 3.2, 5.0]])
    classes_pred = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    predict = np.array([knn.classify(x) for x in X_pred])
    print(f'Clases predichas -> {predict}')
    print(f'Clases reales -> {classes_pred}')
    # METRICAS
    confusion_matrix = cm.confusion_matrix(classes_pred, predict)
    funcs = [cm.accuracy, cm.recall, cm.fp_rate, cm.specificity, cm.precission, cm.f1_score, cm.kappa]
    metrics = [func(confusion_matrix) for func in funcs]
    print(f'CCR -> {metrics[0]}')
    print(f'Recall -> {metrics[1][0]}')
    print(f'FP Rate -> {metrics[2][0]}')
    print(f'Especifidad -> {metrics[3][0]}')
    print(f'Precision -> {metrics[4][0]}')
    print(f'F1 Score -> {metrics[5][0]}')

def problema_3():
    # Comenzamos cargando el conjunto de entrenamiento
    df = pd.read_csv('./l3p3_train.csv')
    # Obtenemos la matriz con las características de cada patrón
    X = np.array(df[['x1', 'x2', 'x3']].values)
    # Obtenemos el vector con las clases de cada patrón
    classes = np.array(df['clase'].values)
    # Generamos el primer clasificador con k = 1
    knn_1 = KNN(X, classes, 1, euclidean)
    # Generamos el segundo clasificador con k = 3
    knn_2 = KNN(X, classes, 3, euclidean)
    # Cargamos el conjunto de test
    df_test = pd.read_csv('./l3p3_test.csv')
    # Obtenemos la matriz con las características de cada patrón
    X_test = np.array(df_test[['x1', 'x2', 'x3']].values)
    # Obtenemos el vector de clases
    classes_test = np.array(df_test['clase'].values)
    # Calculamos la predicción para cada patrón del conjunto de test
    predict_1 = np.array([knn_1.classify(x) for x in X_test])
    predict_2 = np.array([knn_2.classify(x) for x in X_test])
    # Mostramos los resultados de las predicciones
    print(f'Clases predichas con k = 1 -> {predict_1}')
    print(f'Clases predichas con k = 3 -> {predict_2}')
    print(f'Clases reales -> {classes_test}')
    # Calculamos las métricas de rendimiento de cada clasificador
    funcs = [cm.accuracy, cm.recall, cm.fp_rate, cm.specificity, cm.precission, cm.f1_score, cm.kappa]
    # Clasificador con k = 1
    confusion_matrix_1 = cm.confusion_matrix(classes_test, predict_1)
    metrics_1 = [func(confusion_matrix_1) for func in funcs]
    print(f"----- CLASIFICADOR CON k = 1 -----")
    res_df_1 = pd.DataFrame({"CCR":metrics_1[0], "Recall":metrics_1[1], "Fp Rate":metrics_1[2],
                            "Especifidad":metrics_1[3], "Precision":metrics_1[4], "F1 Score":metrics_1[5],
                            "Kappa":metrics_1[6]}, index = list(range(1,4)))
    print(f"Metricas de clasificación -> \n{res_df_1}")
    # Clasificador con k = 3
    confusion_matrix_2 = cm.confusion_matrix(classes_test, predict_2)
    metrics_2 = [func(confusion_matrix_2) for func in funcs]
    print(f"\n\n----- CLASIFICADOR CON k = 3 -----")
    res_df_2 = pd.DataFrame({"CCR":metrics_2[0], "Recall":metrics_2[1], "Fp Rate":metrics_2[2],
                            "Especifidad":metrics_2[3], "Precision":metrics_2[4], "F1 Score":metrics_2[5],
                            "Kappa":metrics_2[6]}, index = list(range(1,4)))
    print(f"Metricas de clasificación -> \n{res_df_2}")

def problema_2():
    # Definimos los vectores de los coeficientes beta
    beta_1 = np.array([541.0741, -33.121, -10.2824, -38.5734, -90.4347])
    beta_2 = np.array([501.1562, -26.3497, -24.2689, -10.7613, -130.0915])
    # Definimos la matriz de características
    X = np.array([[4.6, 3.2, 1.4, 0.2],
                [5.3, 3.7, 1.5, 0.2],
                [5.7, 4.4, 1.5, 0.4],
                [5.0, 3.5, 1.6, 0.6],
                [5.5, 2.5, 4.0, 1.3],
                [5.7, 3.0, 4.2, 1.2],
                [5.7, 2.8, 4.2, 1.3],
                [5.8, 2.7, 5.1, 1.9],
                [6.3, 2.5, 5.0, 1.9],
                [5.9, 3.0, 5.1, 1.8]])
    # Definimos el vector de clases
    classes = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    # Calculamos f1 sobre todos los puntos
    f_1 = beta_1[0] + np.dot(beta_1[1:], np.transpose(X))
    # Calculamos f2 sobre todos los puntos
    f_2 = beta_2[0] + np.dot(beta_2[1:], np.transpose(X))
    # Calculamos la exponencial de dichas funciones
    exp_f1 = np.exp(f_1)
    exp_f2 = np.exp(f_2)
    # Calculamos la probabilidad de pertenencia a cada clase
    sum_exp = exp_f1 + exp_f2
    prob_y1 = exp_f1 / (1 + sum_exp)
    prob_y2 = exp_f2 / (1 + sum_exp)
    prob_y3 = 1 - prob_y1 - prob_y2
    list_prob = list(zip(prob_y1, prob_y2, prob_y3))
    classes_pred = np.array([tup.index(max(tup)) for tup in list_prob]) + 1
    # Mostramos los resultados obtenidos en un DataFrame
    res_df = pd.DataFrame({"exp(f1)":exp_f1, "exp(f2)":exp_f2, "sum(exp)":sum_exp,
                            "p(Y = 1|X)":prob_y1, "p(Y = 2|X)":prob_y2, "p(Y = 3|X)":prob_y3,
                            "Clase Predicha":classes_pred, "Clase Real":classes}, index = list(range(1,11)))
    print(res_df)
    # Calculamos las métricas de rendimiento
    confusion_matrix = cm.confusion_matrix(classes, classes_pred)
    funcs = [cm.accuracy, cm.recall, cm.fp_rate, cm.specificity, cm.precission, cm.f1_score, cm.kappa]
    metrics = [func(confusion_matrix) for func in funcs]
    res_df = pd.DataFrame({"CCR":metrics[0], "Recall":metrics[1], "Fp Rate":metrics[2],
                            "Especifidad":metrics[3], "Precision":metrics[4], "F1 Score":metrics[5],
                            "Kappa":metrics[6]}, index = list(range(1,4)))
    print(f"\n\nMetricas de clasificación -> \n{res_df}")

def problema_1():
    # Cargamos el conjunto de entrnamiento
    df_train = pd.read_csv('./l3p1_train.csv')
    # Obtenemos los valores X1, X2 e Y
    Y = np.array(df_train['y'].values)
    X1 = np.array(df_train['x1'].values)
    X2 = np.array(df_train['x2'].values)
    # Generamos los Regresores Lineales Simples
    linear_1 = SimpleLinearRegressor(X1, Y)
    linear_2 = SimpleLinearRegressor(X2, Y)
    linear_3 = MultipleLinearRegressor(np.row_stack((X1, X2)), Y)
    # Obtenemos para cada regresor simple sus parámetros beta_0 y beta_1
    b0_1, b1_1 = linear_1.get_parameters()
    b0_2, b1_2 = linear_2.get_parameters()
    b0_3, b1_3, b2_3 = linear_3.get_parameters()
    # Mostramos por pantalla la ecuación de cada regresor
    print(f'Regresor para X1 -> Y = {b0_1} + {b1_1}*X1')
    print(f'Regresor para X2 -> Y = {b0_2} + {b1_2}*X2')
    print(f'Regresor para X1 y X2 -> Y = {b0_3} + {b1_3}*X1 + {b2_3}*X2')
    # Cargamos el conjunto de test
    df_test = pd.read_csv('./l3p1_test.csv')
    Y_test = np.array(df_test['y'].values)
    X1_test = np.array(df_test['x1'].values)
    X2_test = np.array(df_test['x2'].values) 
    # Obtenemos las predicciones de cada regresor
    Y_pred_1 = linear_1.predict(X1_test)
    Y_pred_2 = linear_2.predict(X2_test)
    Y_pred_3 = linear_3.predict(np.row_stack((X1_test, X2_test)))
    # Obtenemos las métricas de cada regresor
    funcs = [rm.mean_absolute_error, rm.mean_squared_error, rm.root_mean_squared_error, rm.coef_det]
    Ys = [Y_pred_1, Y_pred_2, Y_pred_3]
    metrics = [list(map(func, [Y_test] * len(Ys), Ys)) for func in funcs]
    # Mostramos los resultados como un dataFrame
    met = ["M1", "M2", "M3"]
    res = pd.DataFrame({"MAE":metrics[0], "MSE":metrics[1], "RMSE":metrics[2], "R^2":metrics[3]}, index = met)
    print(f'Métricas de regresión -> \n {res}')
    print(sklearn.metrics.r2_score(Y_test, Y_pred_3))


if __name__ == "__main__":
    #prueba_1()
    problema_1()
    #prueba_2()
    #problema_2()
    #prueba_3()
    #problema_3()
