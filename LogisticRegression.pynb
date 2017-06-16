import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import normalize,scale
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

class LR (object) :
    def __init__ (self , alpha , Weight , epochs , data=[] , target=[]) :
        self.alpha = alpha
        self.epochs = epochs 
        self.Weight = Weight
        
    def NetInput (self , Data , Target ) :
        NetIn = []
        for i in range (Standered_Data.shape[0]) :
            H = np.dot (Weight , Data) # Z = W.X - equation
            T = append.NetIn(H) 
          
        return NetIn
    
            
    
    def ActivationFn (self , Data , Target ):
        #We can also named it "Prediction Function" 
        Predict = 1 / (1+np.exp(self.NetIn))
        
        return Predict
    
    def UpdateWeight (self ,Target , Data) :
        Error = 0
        for i in range (Data.shape[1]) :
            for j in range (Data.shape[0]) :
                w = (self.Predict[j] - Target[j])*Data[i]
                
                w= Error
        return Error
    
    def CostFn (self , Target) :
        pass
        
    def Fit (self) :
        pass
        
       
       
       
      #upload data  
       Iris = datasets.load_iris() # Iris dataset 
Iris_Data = Iris.data       # extract Iris data
Iris_Target = Iris.target   # extract Iris targets
data = Iris_Data[:100,:]     # Extract 2 class data instead of the 3 class
target = Iris_Target[:100]   # Extract 2 class target instead of the 3 class

#split data to train and test by using "sklearn.cross_validation import train_test_split "
X_train, X_test, y_train, y_test = train_test_split(Standered_Data, target , test_size=0.33 , random_state =100 ) #split_data
train = np.array(zip(X_train,y_train))
test = np.array(zip(X_test, y_test))
# Some values which important to normalize the data 
MeanV = np.mean (data) # The avarage value of data list
MaxV= np.max (data) # The maximum value of data list
MinV = np.min (data) # the minimum value of data list
