# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from operator import itemgetter
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    feature_sel = SelectPercentile(f_regression, percentile = k)
    a=feature_sel.fit_transform(X,y)
    #print a
    column = data.columns
    feature_ind = feature_sel.get_support(indices = True)
    feature_score = feature_sel.scores_[feature_ind]
    #feature_ind = [column[x] for x in feature_sel.get_support(indices=True) if x]
    #print feature_ind
    #print feature_score
    feature_ind_score = zip(feature_score,feature_ind)
    feature_ind_score.sort(key=itemgetter(0), reverse=True)
    #print feature_ind_score
    k_score,k_ind=zip(*feature_ind_score)
    #print k_ind
    k_features=[]
    for x in k_ind:
        k_features.append(column[x])

    #fis_sort = sorted(feature_ind_score.values(), key=operator.itemgetter(0),reverse=True)
    #k_ind = [feature_ind for feature_ind in feature_ind_score]
    #print fis_sort
    return k_features
