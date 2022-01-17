from TestModel import TestModel
import pandas as pd
from model_definitions import m1, m2, m3, SelectTransformer, prep_df

df_train = prep_df( pd.read_csv('train.csv'))
df_test = prep_df(pd.read_csv('test.csv'))



t = SelectTransformer(features = ['Parch'],
                      cat_features = ['Parch'])


t.fit(df_train)


x = t.transform(df_test)


