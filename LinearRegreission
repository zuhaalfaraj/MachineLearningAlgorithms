import csv
import numpy as np
import urllib
from sklearn import datasets
import matplotlib.pyplot as plt

class Multiple_linear_regression (object) :
    def __init__ (self , alpha ,itera , theta=[] ):
        self.alpha = alpha
        self.itera = itera 
        self.theta = theta
    
    def hypothises (self , X , theta ) :
        yPrediction = []
        for i in range (X.shape[0]):
            Prediction = np.dot(self.theta, X[:,i])
            P = yPrediction.append(Prediction)
            
        return yPrediction
            
    def cost (self , Y , X):
        costfn = []
        for j in range (Y.shape[0]):
            fn = pow ((yPrediction[j] - Y[j]) ,2 )
            Ovall = costfn.append (fn)    
        return costfn
    def summtion (self , summtionn):
        summtionn = []
        for i in range (Y.shape[0]) :
            k= (yPrediction[i] - Y[i])
            T=summtionn.append(k)
    def Gradient_Descent (self , X , Y , alpha):
        Error = []
        for A in range (Y.shape[0]):
            for Z in range (Y.shape[0]) :
                GD = theta[A][Z]- self.alpha* self.summtionn
            Grad = Error.append (GD)
        

    def Normalization (self , X):
        pass
