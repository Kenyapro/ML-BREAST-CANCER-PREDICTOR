
#Pandas is a Python library specifically designed for working with data sets.
# sklearn, is a fantastic Python library for machine learning. 
#Scikit-learn is an open-source library that provides simple and efficient tools for predictive data analysis.(Linear regression)
#sk-learn (starndard sclaler) Helps prevent features with large variances from dominating the learning process when training the model.

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
#Model

def create_model (data):
    #predictive variable (manipulates to observe its impact on another variable(relationship))
    
    x = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']
    
    #scale the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    #split the data
    x_train, x_test, y_train, y_test = train_test_split(
        
        x, y, test_size = 0.2, random_state = 42 
    )
    
    #train the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    #test model
    y_pred = model.predict(x_test)
    #print( accuracy_score(y_test, y_pred))
    #print('CLASSIFICATION REPORT >>> /n  ', classification_report(y_test, y_pred))
    
    
    return model, scaler
    
    
    
#Clean our data

def get_clean_data() :
    
    data = pd.read_csv('Data/data.csv')
    data = data.drop(['Unnamed: 32', 'id' ] , axis = 1)
    
    #On the diagnosis colunm we want all benign cells to be 0,malignant cells to be 1
    #The map() function applies a given function to each item in an iterable (like a list, tuple, or dictionary).
    #It returns a new iterable (a map object)
    
    data ['diagnosis'] = data ['diagnosis'].map ({ 'M' :1 , 'B' : 0})
    
    return data  

  
def main() :
    
    data = get_clean_data()    
    
    model, scaler = create_model(data)
    
    with open('APP/model.pkl', 'wb') as f :
        pickle.dump(model, f) 
    
    
    
    with open('APP/scaler.pkl', 'wb') as f :
        pickle.dump(scaler, f) 
    
    
    
if __name__ == '__main__' :
    main()