
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd


class TestModel():

    def __init__(self, n_splits):
        
        self.kfold = KFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=1)
        
    
    def evaluate(self, 
                 modelKlass, 
                 model_kwargs,
                 TargetTransformer,
                 target_kwargs,
                 FeatureTransformer,
                 feature_kwargs,
                 prep,
                 df_train, 
                 df_test):
        
        df_train = prep(df_train)
        df_test = prep(df_test)
        ft = FeatureTransformer(**feature_kwargs)
        Xtrain = ft.fit_transform(df_train)
            
        tt = TargetTransformer(**target_kwargs)
        ytrain = tt.fit_transform(df_train)
            
        m = modelKlass(**model_kwargs)
            
        m.fit(Xtrain, ytrain)
            
        Xtest = ft.transform(df_test)
        pred = m.predict(Xtest)
        
        return pred
        
        
        
        
    def _test(self,
              modelKlass, 
              model_kwargs,
              TargetTransformer,
              target_kwargs,
              FeatureTransformer,
              feature_kwargs,
              prep,
              df_train, 
              df_test):
        
        pred = self.evaluate(modelKlass = modelKlass, 
                             model_kwargs =model_kwargs,
                             TargetTransformer = TargetTransformer,
                             target_kwargs =target_kwargs,
                             FeatureTransformer = FeatureTransformer,
                             feature_kwargs = feature_kwargs,
                             prep = prep,
                             df_train = df_train, 
                             df_test= df_test)
        
        tt = TargetTransformer(**target_kwargs)
        tt.fit_transform(df_train)
        ytest = tt.transform(df_test)
        
        
        return pred, ytest
        
        
        
    def test(self, 
             model_name,
             modelKlass, 
             model_kwargs,
             TargetTransformer,
             target_kwargs,
             FeatureTransformer,
             feature_kwargs,
             prep,
             df):
            
        preds = list()
        ytests = list()
        for tr_index, tst_index in self.kfold.split(df):
            
            df_train = df.iloc[tr_index, :]
            df_test = df.iloc[tst_index, :]
            
            pred, ytest = self._test(modelKlass = modelKlass, 
                                     model_kwargs = model_kwargs,
                                     TargetTransformer = TargetTransformer,
                                     target_kwargs = target_kwargs,
                                     FeatureTransformer = FeatureTransformer,
                                     feature_kwargs = feature_kwargs,
                                     prep = prep,
                                     df_train = df_train,
                                     df_test = df_test)
            preds.append(pred)
            ytests.append(ytest)
            
        all_pred = np.concatenate(preds)
        all_ytest = np.concatenate(ytests)
            
        
        p_test, r_test, f_test, s_test = precision_recall_fscore_support(all_ytest, 
                                                                         all_pred, 
                                                                         average = 'binary')
        acc_test = accuracy_score(all_ytest, all_pred)
        
        pred_train, ytrain  = self._test(modelKlass = modelKlass, 
                                         model_kwargs = model_kwargs,
                                         TargetTransformer = TargetTransformer,
                                         target_kwargs = target_kwargs,
                                         FeatureTransformer = FeatureTransformer,
                                         feature_kwargs = feature_kwargs,
                                         prep = prep,
                                         df_train = df,
                                         df_test = df)
        
        p_train, r_train, f_train, s_train = precision_recall_fscore_support( ytrain, 
                                                                             pred_train, 
                                                                             average = 'binary')
        acc_train = accuracy_score(ytrain, pred_train)
        
        res = [{
                  'p_test' : p_test,
                  'r_test' : r_test,
                  'f_test' : f_test,
                  'acc_test' : acc_test,
                  'p_train' : p_train,
                  'r_train' : r_train,
                  'f_train' : f_train,
                  'acc_train' : acc_train,
                  'model_name' : model_name,
              }]
        
        return pd.DataFrame(res)
    
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

class TestModelHolder():

    def __init__(self, 
                 n_splits,
                 target_feature):
        
        self.kfold = KFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=1)
        self.target_feature = target_feature
    
    def evaluate(self, 
                 modelKlass, 
                 model_kwargs,
                 df_train, 
                 df_test):
        
        m = modelKlass(**model_kwargs)
            
        m.fit(df_train)
            
        pred = m.predict(df_test)
        
        return pred
        
        
    def _test(self,
              modelKlass, 
              model_kwargs,
              df_train, 
              df_test):
        
        pred = self.evaluate(modelKlass = modelKlass, 
                             model_kwargs =model_kwargs,
                             df_train = df_train, 
                             df_test= df_test)
        
        ytest = df_test[self.target_feature].values        
        
        return pred, ytest
        
        
        
    def test(self, 
             model_name,
             modelKlass, 
             model_kwargs,
             df):
            
        preds = list()
        ytests = list()
        for tr_index, tst_index in self.kfold.split(df):
            
            df_train = df.iloc[tr_index, :]
            df_test = df.iloc[tst_index, :]
            
            pred, ytest = self._test(modelKlass = modelKlass, 
                                     model_kwargs = model_kwargs,
                                     df_train = df_train,
                                     df_test = df_test)
            preds.append(pred)
            ytests.append(ytest)
            
        all_pred = np.concatenate(preds)
        all_ytest = np.concatenate(ytests)
            
        
        p_test, r_test, f_test, s_test = precision_recall_fscore_support(all_ytest, 
                                                                         all_pred, 
                                                                         average = 'binary')
        acc_test = accuracy_score(all_ytest, all_pred)
        
        pred_train, ytrain  = self._test(modelKlass = modelKlass, 
                                         model_kwargs = model_kwargs,
                                         df_train = df,
                                         df_test = df)
        
        p_train, r_train, f_train, s_train = precision_recall_fscore_support( ytrain, 
                                                                             pred_train, 
                                                                             average = 'binary')
        acc_train = accuracy_score(ytrain, pred_train)
        
        res = [{
                  'p_test' : p_test,
                  'r_test' : r_test,
                  'f_test' : f_test,
                  'acc_test' : acc_test,
                  'p_train' : p_train,
                  'r_train' : r_train,
                  'f_train' : f_train,
                  'acc_train' : acc_train,
                  'model_name' : model_name,
              }]
        
        return pd.DataFrame(res)
    
