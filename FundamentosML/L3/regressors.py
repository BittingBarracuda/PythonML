import numpy as np
import pymatrix as pym

def scale_values(X, min_x, max_x):
        return (X - min_x) / (max_x - min_x)

class SimpleLinearRegressor:

    def __init__(self, X, Y):
        self.__min_x = np.min(X)
        self.__max_x = np.max(X)
        self.X = scale_values(X, self.__min_x, self.__max_x)
        self.Y = Y
        self.b0, self.b1 = self.__max_ver_parameters()

    def __max_ver_parameters(self):
        # Obtenemos beta1 = Sxy / Sx^2
        mean_x, mean_y = np.mean(self.X), np.mean(self.Y)
        dif_x, dif_y = self.X - mean_x, self.Y - mean_y
        beta1 = np.sum(dif_y * dif_x) / np.sum(dif_x * dif_x)
        beta0 = mean_y - beta1*mean_x
        return beta0, beta1

    def get_parameters(self):
        return self.b0, self.b1

    def predict(self, x_t):
        x_aux = scale_values(x_t, self.__min_x, self.__max_x)
        return self.b0 + self.b1 * x_aux
    

class MultipleLinearRegressor:
    def __init__(self, X, Y):
        # Entendemos X como una matriz donde cada fila es una variable independiente
        self.mins_X = [np.min(Xi) for Xi in X]
        self.maxs_X = [np.max(Xi) for Xi in X]
        aux = [(X[i], self.mins_X[i], self.maxs_X[i]) for i in range(len(X))]
        self.X = np.array([scale_values(Xi, min_Xi, max_Xi) for (Xi, min_Xi, max_Xi) in aux])
        self.Y = Y
        self.cov = self.__get_cov_matrix()
        self.betas = self.__max_ver_parameters()
        
    def __get_cov_matrix(self):
        aux = np.row_stack((self.Y, self.X))
        return np.cov(aux)
    
    def __max_ver_parameters(self):
        # Obtenemos la matriz adjunta mediante pymatrix
        aux_pym = pym.Matrix.from_list(self.cov)
        aux_adjoint = aux_pym.adjoint()
        # Obtenemos beta1...betak usando la matriz adjunta
        res = np.zeros((1, len(self.X) + 1)).flatten()
        for i in range(1, len(res)):
            res[i] = -aux_adjoint[0][i] / aux_adjoint[0][0]
        # Obtain beta_0
        X_means = np.mean(self.X, axis = 1)
        Y_mean = np.mean(self.Y)
        # Finalmente obtenemos beta0
        res[0] = Y_mean - np.dot(res[1:].flatten(), X_means)
        return res

    def get_parameters(self):
        return self.betas
    
    def predict(self, X):
        aux = [(X[i], self.mins_X[i], self.maxs_X[i]) for i in range(len(X))]
        aux_X = [scale_values(Xi, min_Xi, max_Xi) for (Xi, min_Xi, max_Xi) in aux]
        return self.betas[0] + np.dot(self.betas[1:], aux_X)