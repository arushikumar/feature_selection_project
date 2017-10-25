# Default imports
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Your solution code here
def rf_rfe(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    cols = X.columns
    #print cols
    rf = RandomForestClassifier()
    rfe = RFE(rf, len(cols)/2)
    rfe = rfe.fit(X,y)
    #print rfe.support_
    features = cols[rfe.support_]
    #features = [cols[x] for x in rfe.support_ if x==True]
    return features.tolist()
