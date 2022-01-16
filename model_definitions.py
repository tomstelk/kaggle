from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import numpy as np
import numpy.ma as ma

class SelectTransformer():

    def __init__(self, features):
        self.features = features
        
    def fit(self, df):
        return self
    
    def transform(self, df):
        return df[self.features].values
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)
    
class SingleSelectTransformer():

    def __init__(self, feature):
        self.feature = feature
        
    def fit(self, df):
        return self
    
    def transform(self, df):
        return df[self.feature].values
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)


def prep_df(df):
    
    out = deepcopy(df)
    out['SexNum'] = [1 if s =='male' else 0 for s in out['Sex'].values]
    a = out['Age'].values
    out['AgeFilled'] = np.where(np.isnan(a), ma.array(a, mask=np.isnan(a)).mean(), a)    
    out['Cabin'] = out['Cabin'].map(str)
    
    return out

f1s = ['Pclass',
       'SexNum',
       'AgeFilled',
       'SibSp',
       'Parch']

m1 = {'prep': prep_df,
      'model_name' : '|'.join(f1s),
      'modelKlass':  LogisticRegression,
      'model_kwargs': dict(),
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : list(f1s)},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}