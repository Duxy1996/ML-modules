import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

pd.options.mode.chained_assignment = None

corpus = pd.read_csv('../corpus.csv')

predictors = corpus[['incomeperperson','employrate','suicideDanger','alcconsumption','femaleemployrate','lifeexpectancy']]

predictors.replace(' ', np.nan, inplace=True)
predictors = predictors.dropna()

predictors['incomeperperson']  = preprocessing.scale(predictors['incomeperperson'].astype('float64'))
predictors['employrate']       = preprocessing.scale(predictors['employrate'].astype('float64'))
predictors['alcconsumption']   = preprocessing.scale(predictors['alcconsumption'].astype('float64'))
predictors['femaleemployrate'] = preprocessing.scale(predictors['femaleemployrate'].astype('float64'))
predictors['lifeexpectancy']   = preprocessing.scale(predictors['lifeexpectancy'].astype('float64'))

target     = predictors[['suicideDanger']]
predictors = predictors[['incomeperperson','employrate','alcconsumption','femaleemployrate','lifeexpectancy']]

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=.15, random_state = 0)

clusters=range(1,10)
meandist = []
loss     = []

for k in clusters:
  model=KMeans(n_clusters=k)
  model.fit(pred_train)
  clusassign=model.predict(pred_train)
  meandist.append(sum(np.min(cdist(pred_train, model.cluster_centers_, 'euclidean'), axis=1)) 
  / pred_train.shape[0])
  if len(meandist) <= 1:
    loss.append(1)
  else:
    loss.append(meandist[-2]/meandist[-1])


print(meandist)
print(loss)

plt.plot(clusters, meandist)
plt.plot(clusters, loss)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()

model3=KMeans(n_clusters=6)
model3.fit(pred_train)
clusassign=model3.predict(pred_train)

pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(pred_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()
