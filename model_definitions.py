from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import numpy as np
import numpy.ma as ma
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats


class ModelHolder():
    
    
    def __init__(self,
                 modelKlass, 
                 model_kwargs,
                 FeatureTransformer,
                 feature_kwargs,
                 TargetTransformer,
                 target_kwargs,
                 prep,
                 **other_stuff):
        

        self.model = modelKlass(**model_kwargs)
        self.feature_transformer = FeatureTransformer(**feature_kwargs)
        self.target_transform = TargetTransformer(**target_kwargs)
        self.prep = prep
        
    def fit(self, df):
        dfp = self.prep(df)
        X = self.feature_transformer.fit_transform(dfp)
        y = self.target_transform.fit_transform(dfp)
        
        self.model.fit(X, y)
        
        return self
    
    def predict(self, df):
        dfp = self.prep(df)
        X = self.feature_transformer.transform(dfp)
        return self.model.predict(X)
    
    def predict_proba(self, df):
        dfp = self.prep(df)
        X = self.feature_transformer.transform(dfp)
        return self.model.predict_proba(X)


class CatCoder():
    
    
    def __init__(self, 
                 features,
                 fix_missing = True):
        
        self.features = features
        self.onehot = OneHotEncoder()
        self.labellers = [ LabelEncoder() for _ in features]
        self.fix_missing = fix_missing
        
    def fit(self, df):
        
        Xl = list()
        for i, f in enumerate(self.features):
            x = list(map(str, df[f].values))
            Xl.append(self.labellers[i].fit_transform(x))
        Xl = np.array(Xl).transpose()

        self.onehot.fit(Xl)

        return self
    
    def _fix_missing(self, labeller, x):
        
        classes = set(labeller.classes_)
        
        if all(xi in classes for xi in x):
            return x
        
        mod = stats.mode(x).mode[0]
        corrected = [xi if xi in classes else mod for xi in x ]
        
        return corrected
        
        
    
    def transform(self, df):
        Xl = list()
        
        for i, f in enumerate(self.features):
            x = list(map(str, df[f].values))
            
            labelleri = self.labellers[i]
            if self.fix_missing:
                x = self._fix_missing(labelleri, x)
            
            Xl.append(labelleri.transform(x))
        Xl = np.array(Xl).transpose()
        return self.onehot.transform(Xl).toarray()
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)
        

class SelectTransformer():

    def __init__(self, 
                 features,
                 cat_features = None,
                 standardise = True):
        
        self.features = features
        if cat_features:
            self.cat_coder = CatCoder(features = cat_features)
        else:
            self.cat_coder = None
            
        if standardise:
            self.standardiser = StandardScaler()
        
    def fit(self, df):
        if self.cat_coder:
            self.cat_coder.fit(df)
            
        if self.standardiser:
            X = self._make_features(df)
            self.standardiser.fit(X)

        return self
    
    def _make_features(self, df):
        X1 = df[self.features].values
        if self.cat_coder:
            X2 = self.cat_coder.transform(df)
            return np.concatenate([X1, X2], axis = 1)
        else:
            return X1
    
    def transform(self, df):
        X = self._make_features(df)

        if self.standardiser:
            return self.standardiser.transform(X)
        else:
            return X
        
    
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



def hist_group_values(x, num_groups):
    
    q = [100*i/num_groups for i in range(num_groups+1)]
    p = np.percentile(x, q)
    
    for xi in x:
        for i, pi in enumerate(p):
            if xi<=pi:
                yield i
                break
    
def prep_df(df):
    
    out = deepcopy(df)
    out['SexNum'] = [1 if s =='male' else 0 for s in out['Sex'].values]
    a = out['Age'].values
    out['AgeFilled'] = np.where(np.isnan(a), ma.array(a, mask=np.isnan(a)).mean(), a)    
    
    
    f= out['Fare'].values
    out['FareFilled'] = np.where(np.isnan(f), ma.array(f, mask=np.isnan(f)).mean(), f)    
    out['Cabin'] = out['Cabin'].map(str)
    
    out['CabinLetter'] = [c[0] for c in out['Cabin'].values]  
    x = out['AgeFilled'].values
    out['AgeGrouped'] = ['a{}'.format(gai) for gai in hist_group_values(x, 19)]
    return out

#############################################
#MODEL  1
f1s = ['SexNum',
       'AgeFilled']
catf1s = ['Pclass', 'Embarked']

m1 = {'prep': prep_df,
      'model_name' : 'LogReg-' + '_'.join(f1s+catf1s) ,
      'modelKlass':  LogisticRegression,
      'model_kwargs': dict(),
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f1s, 'cat_features':catf1s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################




#############################################
#MODEL  2
f2s = ['SexNum', 
       'AgeFilled',
       'SibSp',
       'Parch',
       'FareFilled']
catf2s = ['Pclass', 'Embarked']

m2 = {'prep': prep_df,
      'model_name' : 'SVC-' + '_'.join(f2s+catf2s),
      'modelKlass':  SVC,
      'model_kwargs': {'probability' : True},
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f2s, 'cat_features':catf2s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################






#############################################
#MODEL  3
f3s = ['SexNum',
       'FareFilled']
catf3s = ['Pclass']

m3 = {'prep': prep_df,
      'model_name' : 'KNN-' + '_'.join(f3s+catf3s),
      'modelKlass':  KNeighborsClassifier,
      'model_kwargs': {'weights' : 'distance', 'n_neighbors' : 50},
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f3s, 'cat_features':catf3s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################





#############################################
#MODEL  4
f4s = ['SexNum',
       'AgeFilled',
       'Pclass']
catf4s = ['SibSp', 'Parch']

m4 = {'prep': prep_df,
      'model_name' : 'LogReg-' + '_'.join(f4s+catf4s) ,
      'modelKlass':  LogisticRegression,
      'model_kwargs': dict(),
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f4s, 'cat_features':catf4s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################





#############################################
#MODEL  5
f5s = ['SexNum', 
       'AgeFilled',
       'Pclass',
       'Parch']
catf5s = ['Embarked']

m5 = {'prep': prep_df,
      'model_name' : 'SVC-' + '_'.join(f5s+catf5s),
      'modelKlass':  SVC,
      'model_kwargs': {'probability' : True},
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f5s, 'cat_features':catf5s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################



#############################################
#MODEL  6
f6s = ['SexNum',
       'FareFilled',
       'Pclass']
catf6s = []

m6= {'prep': prep_df,
      'model_name' : 'KNN-' + '_'.join(f6s+catf6s),
      'modelKlass':  KNeighborsClassifier,
      'model_kwargs': {'weights' : 'distance', 'n_neighbors' : 50},
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f6s, 'cat_features':catf6s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################


