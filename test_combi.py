from itertools import combinations

all_features =  ['Pclass','SexNum','AgeFilled','SibSp', 'Parch', 'Fare',  'Embarked'] #'Cabin',
all_cat_features = ['Embarked'] #'Cabin',
all_maybe_cat_features = ['Pclass', 'SibSp', 'Parch']

n = 0

def split_maybes(mfs):
    
    n = len(mfs)
    if n==0:
        return list()
    
    for i in range(0, n+1):
        for cmfs in combinations(mfs, i):
            yield list(cmfs)


for i in range(7, 8):
    for fs in combinations(all_features, i):
    
        maybe_cat_features = [ fi for fi in fs if fi in all_maybe_cat_features]
        
        for cfs in split_maybes(maybe_cat_features):
            
            ncfs = [mcf for mcf in maybe_cat_features if mcf not in cfs]
            
            cat_features = [ fi for fi in fs if fi in all_cat_features] + cfs
            non_cat_features = [ fi for fi in fs if (fi not in all_cat_features) and (fi not in all_maybe_cat_features)] + ncfs
            
            
            print('---------')
            print(cat_features, non_cat_features)


