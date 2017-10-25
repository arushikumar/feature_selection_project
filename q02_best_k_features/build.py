# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(data, k=20):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    feature_sel = SelectPercentile(f_regression, percentile = k)
    feature_sel.fit_transform(X,y)
    column = data.columns
    feature_ind = [column[x] for x in feature_sel.get_support(indices=True) if x]
    return feature_ind
