import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def prepare_num(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Emb')
    df_pcl = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num

df_main = pd.read_csv('data/train.csv')


df_prep_x = df_main.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

df_prep_y = df_main['Survived']

df_prep_x_num = prepare_num(df_prep_x)


df_prep_x_num = df_prep_x_num.fillna(df_prep_x_num.median())


X_train, X_test, Y_train, Y_test = train_test_split(df_prep_x_num, df_prep_y, test_size= 0.15, random_state= 42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size= 0.15, random_state= 42)

df_main