import numpy as np
from random import choice
from collections import Counter

class KNN:
    def __init__(self, X, classes, k, dist):
        if(len(X) != len(classes)):
            raise ValueError("Data matrix (X) and class vector (classes) must have the same number of rows")
        self.X = X
        self.class_vector = classes
        self.classes = np.unique(classes)
        self.k = k
        self.dist = dist
    
    def classify(self, x):
        # Calculamos las distancias a todos los puntos
        distances = self.dist(x, self.X)
        # Ordenamos en orden creciente las distancias
        distances = sorted(list(zip(distances, self.class_vector)), key = lambda x: x[0])
        # Nos quedamos con los k primeros patrones
        k_near = distances[0:self.k]
        candidates = [x[1] for x in k_near]
        # Finalmente elegimos la clase más común de entre las k seleccionadas
        aux = Counter(candidates).most_common()
        # En caso de que existan clases igual de comunes, elegimos aleatoriamente
        max = aux[0][1]
        aux_maxes = [candidate for candidate in aux if candidate[1] == max]
        return choice(aux_maxes)[0]
