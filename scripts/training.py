import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
#%%
def age_classifier(age):
    if (age<0):
        return 'missing'
    elif (age>=0 and age<=12):
        return 'a_infant'
    elif (age>12 and age <=18):
        return 'b_teenager'
    elif (age>18 and age<=35):
        return 'c_young'
    elif (age>35 and age <=60):
        return 'd_adult'
    elif (age>60 and age <= 80):
        return 'e_senior'
    else: return 'f_supersenior'

#%%
def data_clean(df_train):
    df_train['Sex'] = df_train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace = True)
    
    df_train['Age'].fillna(-1, inplace = True)
    ages = df_train[['PassengerId', 'Age']].copy()
    ages['Age'] = ages['Age'].apply(age_classifier)
    '''
    sibsp = df_train[['PassengerId', 'Survived', 'SibSp']].copy()
    sibsp['SibSp'] = sibsp['SibSp'].apply(lambda x: 1 if (x == 1 or x == 2) else 0)
    
    parch = df_train[['PassengerId', 'Survived', 'Parch']].copy()
    parch['Parch'] = parch['Parch'].apply(lambda x: 1 if (x>=1 and x<=3) else 0)
    '''
    df_train['Family'] = df_train['SibSp'] + df_train['Parch']
    df_train['Family'] = df_train['Family'].apply(lambda x: 1 if (x>=1 and x<=3) else 0)
    
    df_train['Embarked'].fillna('S', inplace = True)
    df_train['Fare'].fillna(df_train['Fare'].mean(), inplace=True)
    df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix = 'Port')], axis=1)
    df_train = pd.concat([df_train, pd.get_dummies(df_train['Pclass'], prefix = 'Class')], axis=1)
    df_train = pd.concat([df_train, pd.get_dummies(ages['Age'])], axis=1)
    df_train.drop(['Pclass', 'Embarked', 'Age', 'SibSp', 'Parch'], axis=1, inplace=True)
    df_train.drop(['Fare', 'Port_Q', 'b_teenager', 'c_young', 'd_adult', 'missing'], axis=1, inplace=True)
    return df_train

#%%
def predictor(clf, df_test):
    df = pd.DataFrame()
    df['PassengerId'] = df_test['PassengerId']
    df['Survived'] = clf.predict(df_test.iloc[:, 1:])
    return df

#%%
df_train = pd.read_csv('../data/train.csv')
df_train = data_clean(df_train)
df_test = data_clean(pd.read_csv('../data/test.csv'))
X_train, X_test, Y_train, Y_test = train_test_split(df_train.iloc[:,2:], df_train['Survived'], test_size = 0.25)

#%%
clf = RandomForestClassifier(n_estimators = 50)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

s


