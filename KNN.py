

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import normalize,scale
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import math



class KNN (object) :
    def __init__ (self , k=3) :
        self.k=k
        
    
    def distance (self, data1 , data2) :
        points = zip(data1, data2)
        D = math.sqrt(sum([pow(a - b, 2) for (a, b) in points]))
        return D
    
    def distance_matrix(self ,test_model ,training_model):
        
         return (training_model, self.distance(test_model, training_model[0]))
                      
    def neighbours_data(self,raining_set, test_model):
        
        distances = [self.distance_matrix(training_model, test_model) for training_model in training_set]
        sorted_distances = sorted(distances, key=itemgetter(1))
        sorted_training_model = [tuple[0] for tuple in sorted_distances]
        return sorted_training_model[:k]
                      
                    
    def vote(self , neighbours):
        
    
        classes = [neighbour[1] for neighbour in neighbours]
        count = Counter(classes)
        return count.most_common()[0][0]

    def predict (self) :
        predictions = []
                      
    def fit (self,data , target) :
        
        
        for x in range(len(X_test)):
                
                neighbours = neighbours(y_train, X_test[x][0], self.k)
                majority_vote = self.vote(neighbours)
                self.predictions.append(majority_vote)
        return self.predictions
