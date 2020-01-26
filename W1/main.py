import pandas as pd
import numpy as np
import pydot

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

corpus = pd.read_csv('../corpus.csv')

dic = {}

for i,col in enumerate(corpus.columns): 
  print(i,col)
  dic[i] = col

features = []
feature_sel = 0

while(feature_sel != -1):
  feature_sel = int(input("> "))
  if (feature_sel >= 0):
    features.append(dic[feature_sel])
    print(features)

depth = int(input("Select depth (1-100) > "))

features.append('suicideDanger')
corpus = corpus[features]

corpus.replace(' ', np.nan, inplace=True)
corpus = corpus.dropna()
corpus = corpus.reset_index(drop=True)

features.pop()
X_train = corpus[features]
Y_trian = corpus[['suicideDanger']]

dummy = Y_trian['suicideDanger'].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_train,Y_trian,random_state = 0)

clas = DecisionTreeClassifier(max_depth = depth).fit(X_train, y_train)

pred = clas.predict(X_test)

y_test = y_test['suicideDanger'].values.tolist()

pred  = zip(pred,y_test)
score = clas.score(X_test,y_test)
scoreT = clas.score(X_train,y_train)

print()
print("Test score: ",score)
print()
print("Train score: ",scoreT)
# predict = clas.predict([[70]])
# print(predict)

out = open("./file.dot", 'w')
tree.export_graphviz(clas, out_file=out)
out.close()

(graph,) = pydot.graph_from_dot_file('./file.dot')
graph.write_png('tree.png')



