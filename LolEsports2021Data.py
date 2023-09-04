import pandas as pd
import numpy as np
import csv
import matplotlib as plt
import warnings
warnings.filterwarnings("ignore")

#Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#Splitting Data
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

#Modeling,
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier

game_stats = pd.read_csv("2021LolesportsData.csv")
games = game_stats.copy()

games.drop(columns = ['datacompleteness', 'url', 'date', ], inplace = True)
is_player = games['participantid'] == 100

games = games[is_player]


games = games[['side', 'firstblood','firstdragon', 'firsttower', 'firstbaron','result',
 'golddiffat15', 'csdiffat15', 'xpdiffat15']]
print('Filter Success')
print(games.head(20))
games_copy = games.copy()
X = games_copy.drop('result', axis = 1)
y = games_copy['result']
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3, random_state = 3434)
print(X_train)
print(y_train)
mode_onehot_pipe = Pipeline([
    ('encoder', SimpleImputer(strategy = 'most frequent')),
    ('one hot encoder', OneHotEncoder(handle_unknown='ignore'))])

transformer = ColumnTransformer([
    ('one hot', OneHotEncoder(handle_unknown='ignore'), ['side', 'firstblood','firstdragon', 'firsttower', 'firstbaron',
 'golddiffat15', 'csdiffat15', 'xpdiffat15']),
    
    ], remainder = 'passthrough')

logreg = LogisticRegression(random_state = 3434)
tree = DecisionTreeClassifier(random_state = 3434)
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state = 3434)
svc = LinearSVC(random_state = 3434)
ada = AdaBoostClassifier(random_state = 3434)
grad = GradientBoostingClassifier(random_state = 3434)
xgb = XGBClassifier(verbosity = 0, random_state = 3434)
logreg_pipe = Pipeline([('transformer', transformer), ('logreg', logreg)])
tree_pipe = Pipeline([('transformer', transformer), ('tree', tree)])
knn_pipe = Pipeline([('transformer', transformer), ('knn', knn)])
rf_pipe = Pipeline([('transformer', transformer), ('rf', rf)])
svc_pipe = Pipeline([('transformer', transformer), ('svc', svc)])
ada_pipe = Pipeline([('transformer', transformer), ('ada', ada)])
grad_pipe = Pipeline([('transformer', transformer), ('grad', grad)])
xgb_pipe = Pipeline([('transformer', transformer), ('xgb', xgb)])
print("In progress")
for model in [logreg_pipe, tree_pipe, knn_pipe, rf_pipe, svc_pipe, ada_pipe, grad_pipe, xgb_pipe]:
    model.fit(X_train, y_train)

score_acc = [accuracy_score(y_test, logreg_pipe.predict(X_test)),
             accuracy_score(y_test, tree_pipe.predict(X_test)),
             accuracy_score(y_test, knn_pipe.predict(X_test)),
             accuracy_score(y_test, rf_pipe.predict(X_test)),
             accuracy_score(y_test, svc_pipe.predict(X_test)),
             accuracy_score(y_test, ada_pipe.predict(X_test)),
             accuracy_score(y_test, grad_pipe.predict(X_test)),
             accuracy_score(y_test, xgb_pipe.predict(X_test))]
method_name = ['Logistic Regression', 'Decision Tree Classifier', 'KNN Classifier', 'Random Forest Classifier', 'LinearSVC', 'AdaBoost Classifier', 'Gradient Boosting Classifier', 'XGB Classifier']

acc_summary = pd.DataFrame({'method': method_name, 'accuracy score': score_acc})
print(acc_summary)



