from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import numpy as np
import numpy.ma as ma
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsRegressor



def test_num_str(x):
    try:
        float(x)
        return True
    except:
        return False

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
    
    def _fix_missing_num(self, x, classes):
        
        set_classes = set(classes)
        classes_num_dict = {float(c):c for c in classes}
        x_num = list(map(float, x))

        get_closest = lambda xi : classes_num_dict[min(classes_num_dict.keys(), key = lambda ci: abs(ci-xi))]
        closest = list(map(get_closest, x_num))
        
        corrected = [x if x in set_classes else xc for x, xc in zip(x, closest)]
        
        return corrected
        
    
    def _fix_missing(self, labeller, x):
        
        classes = set(labeller.classes_)
        
        if all(xi in classes for xi in x):
            return x
        
        
        if all (test_num_str(xi) for xi in x):
            corrected = self._fix_missing_num(x, classes)
        else:
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





class KNNImputer():
    
    
    def __init__(self, 
                 impute_col, 
                 out_col,
                 feature_cols,
                 **knn_kwargs):

        self.impute_col = impute_col
        self.out_col = out_col
        self.feature_cols = feature_cols
        self.knn = KNeighborsRegressor(**knn_kwargs)
        
    def fit(self, df):
        
        df = df[~df[self.impute_col].isnull()]
        
        y = df[self.impute_col].values
        X = df[self.feature_cols].values
        
        self.knn.fit(X, y)
        return self
        
    def transform(self, df):

        out = deepcopy(df)
        dfmissing = df[df[self.impute_col].isnull()]
        X = dfmissing[self.feature_cols].values
        
        missing_index = dfmissing.index.values
        missing_values = self.knn.predict(X)
        
        out[self.out_col] = out[self.impute_col].values
        
        for i, v in zip(missing_index, missing_values):
            out.loc[i,self.out_col] = v
            
        return out
            
        
        

def hist_group_values(x, num_groups):
    
    q = [100*i/num_groups for i in range(num_groups+1)]
    p = np.percentile(x, q)
    
    for xi in x:
        for i, pi in enumerate(p):
            if xi<=pi:
                yield i
                break
    
def prep_df(df,
            age_groups = 10, 
            fare_groups = 10,
            k_age_imputer = 10):
    
    out = deepcopy(df)
    out['SexNum'] = [1 if s =='male' else 0 for s in out['Sex'].values]
    
    k = KNNImputer(impute_col = 'Age',
                   out_col= 'AgeFilled',
                   feature_cols =['Pclass',  'SibSp', 'Parch', 'SexNum'],
                   n_neighbors = k_age_imputer)
    k.fit(out)
    
    out = k.transform(out)
    
    
    f= out['Fare'].values
    out['FareFilled'] = np.where(np.isnan(f), ma.array(f, mask=np.isnan(f)).mean(), f)    
    
    xf = out['FareFilled'].values
    out['FareGrouped'] = [str(gfi) for gfi in hist_group_values(xf, fare_groups)]
    
    
    out['Cabin'] = out['Cabin'].map(str)
    out['CabinLetter'] = [c[0] for c in out['Cabin'].values]  
    
    xa = out['AgeFilled'].values
    out['AgeGrouped'] = [str(gai) for gai in hist_group_values(xa, age_groups)]
    
    
    get_title = lambda n : n.split(',')[1].split('.')[0].strip()

    titles = list(map(get_title, out['Name'].values))

    default_title = 'Other'
    titles_keep = {'Master', 'Miss', 'Mr', 'Mrs'}


    out['Title'] = [t if t in titles_keep else default_title for t in titles]
    
    
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


#############################################
#MODEL  7
f7s = ['SexNum',
       'Pclass',
       'SibSp'
       ]
catf7s = ['Embarked',
          'CabinLetter',
          'AgeGrouped']

m7 = {'prep': prep_df,
      'model_name' : 'LogReg-' + '_'.join(f7s+catf7s) ,
      'modelKlass':  LogisticRegression,
      'model_kwargs': dict(),
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f7s, 'cat_features':catf7s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################





#############################################
#MODEL  8
f8s = ['SexNum',
       'AgeFilled',
       'SibSp'
       ]
catf8s = []

m8 = {'prep': prep_df,
      'model_name' : 'GaussianNB-' + '_'.join(f8s+catf8s) ,
      'modelKlass':  GaussianNB,
      'model_kwargs': dict(),
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f8s, 'cat_features':catf8s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################




#############################################
#MODEL  9
f9s = ['SexNum',
       'AgeFilled',
       'SibSp',
       'Parch'
       ]
catf9s = []

m9 = {'prep': prep_df,
      'model_name' : 'QDA-' + '_'.join(f9s+catf9s) ,
      'modelKlass':  QDA,
      'model_kwargs': dict(),
      'FeatureTransformer': SelectTransformer,
      'feature_kwargs': {'features' : f9s, 'cat_features':catf9s},
      'TargetTransformer' : SingleSelectTransformer,
      'target_kwargs' : {'feature' : 'Survived'}}
#############################################



