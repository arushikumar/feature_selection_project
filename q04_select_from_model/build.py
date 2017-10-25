# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    cols = X.columns
    np.random.seed(9)
    #print cols
    rf = RandomForestClassifier()
    rf.fit(X,y)
    model = SelectFromModel(rf, prefit=True)
    #print model.get_support()
    #print rf.get_support()
    features = cols[model.get_support()]
    #features = [cols[x] for x in rfe.support_ if x==True]
    return features.tolist()
