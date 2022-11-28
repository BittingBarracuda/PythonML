import numpy as np

def confusion_matrix(x_real, x_pred):
    # Vamos a hacer una definición general de esta función, es decir, 
    # se podrá utilizar en problemas de clasificación en dos o más clases.
    # Obtenemos las diferentes clases del problema
    classes = np.unique(x_real)
    n = len(classes)
    # Definimos la matriz inicialmente toda a 0s
    matrix = np.zeros(shape = (n, n))
    # Comenzamos a rellenar la matriz
    for i in range(n):
        # Obtenemos la clase actual
        current_class = classes[i]
        # Posiciones donde el valor real vale current_class
        pos1 = np.where(x_real == current_class)
        for j in range(n):
            # Obtenemos las posiciones de la classes[j]
            pos2 = np.where(x_pred == classes[j])
            # Contamos los aciertos/fallos viendo en cuantas posiciones
            # coinciden pos1 y pos2.
            # Si classes[j] == current_class las posiciones donde coinciden
            # pos1 y pos2 son los aciertos
            # Si classes[j] != currenc_class las posiciones donde coinciden
            # pos1 y pos2 son los fallos del modelo
            matrix[i, j] = len(np.intersect1d(pos1, pos2))
    return matrix
            
def true_values(matrix):
    # Los valores correctamente clasificados se encuentran en la diagonal
    # de la matriz de confusión
    return np.diagonal(matrix)    
    
def accuracy(matrix):
    return np.sum(true_values(matrix)) / np.sum(matrix)

def recall(matrix):
    sum_rows = np.sum(matrix, axis = 1)
    return true_values(matrix) / sum_rows

def fp_rate(matrix):
    sum_columns = np.sum(matrix, axis = 0)
    fp = sum_columns - true_values(matrix)
    tn = np.zeros((1, len(matrix))).flatten()
    for i in range(len(matrix)):
        tn[i] = np.sum(np.delete(np.delete(matrix, i, axis = 0), i, axis = 1))
    return fp / (tn + fp)
    
    #sum_columns = np.sum(matrix, axis = 0)
    #sum_rows = np.zeros((1, len(matrix)))
    #for i in range(len(sum_rows)):
    #    sum_rows[i] = np.sum(np.delete(matrix, i, axis = 0))
    #return (sum_columns - true_values(matrix)) / sum_rows.flatten()

def specificity(matrix):
    sum_columns = np.sum(matrix, axis = 0)
    fp = sum_columns - true_values(matrix)
    tn = np.zeros((1, len(matrix))).flatten()
    for i in range(len(matrix)):
        tn[i] = np.sum(np.delete(np.delete(matrix, i, axis = 0), i, axis = 1))
    return tn / (tn + fp)
    
    
    #true_val = true_values(matrix)
    #aux_true = np.zeros((1, len(true_val)))
    #sum_rows = np.zeros((1, len(matrix)))
    #for i in range(len(aux_true)):
    #    aux_true[i] = np.sum(np.delete(true_val, i))
    #    sum_rows[i] = np.sum(np.delete(matrix, i, axis = 0))
    #return aux_true.flatten() / sum_rows.flatten()

def precission(matrix):
    sum_columns = np.sum(matrix, axis = 0)
    return true_values(matrix) / sum_columns

def f1_score(matrix):
    prec = precission(matrix)
    rec = recall(matrix)
    return (2 * prec * rec) / (prec + rec)

def kappa(matrix):
    p0 = accuracy(matrix)
    sum_rows = np.sum(matrix, axis = 1)
    sum_columns = np.sum(matrix, axis = 0)
    pe = (1 / (np.sum(matrix) ** 2)) * np.sum(sum_rows * sum_columns)
    return (p0 - pe) / (1 - pe)