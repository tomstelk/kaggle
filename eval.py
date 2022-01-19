import pandas as pd
from collections import Counter
from copy import deepcopy


        

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

get_title = lambda n : n.split(',')[1].split('.')[0].strip()

titles = list(map(get_title, df_train['Name'].values))

default_title = 'Other'
titles_keep = {'Master', 'Miss', 'Mr', 'Mrs'}


df_train['Title'] = [t if t in titles_keep else default_title for t in titles]