# Suicide risk prevention IV (Clustering)
### Exercise 3 from Machine Learning for Data Analysis

## Introduction

In the past week we tried to improve the suicide risk detection using a decision tree and many others classificators such us random
forest or trees. The article could be found in this git repository here:

[Suicide risk prevention I](https://github.com/Duxy1996/ML-modules/tree/master/W1).
[Suicide risk prevention II](https://github.com/Duxy1996/ML-modules/tree/master/W2).
[Suicide risk prevention III](https://github.com/Duxy1996/ML-modules/tree/master/W3).

The results given were optimistic,vwe could know in which city/zone it is probably to have the suicide problem. The accuracy of the 
classifier was **92%** for the test data.

The parameters used to found this result where: **income, emplyment rate and alcohol consumption**. In this week we are going to analyse the main factors of this problem using the Lasso regresion method to know which parameters are the most significative. Then we will compare the results with the **Random forest most important values for classification**.

## Problem:
The suicide rate is raising in many countries which the problem seemed to be controlled. The main factors of this case are not
well known yet, and this fact is dangerous because we cannot do anything without knowing the problem.

In countries such us United Kindom and EEUU suicides are rising and, now the average is over the 15 people over 100.000 people which
is worrying.

We want to use decision trees to predict in which zones this rate can raise. Using the **GapMinder** dataset we will discover which
zones are over the 15 per 100.000 of suicide rate. Knowing this information wee could make a first approach to the problem and help the
people which is in the risk group.

In this week we want to know which conclusions we could extract from the raw data. Which clusters we can ensure.

## Dataset:
**GapMinder** is a good reference to get data from several countries and make a study to detect people which is in the suicide risk group and help with its problems before one fatality could appear. This dataset has information about different features and targets
which are used to measure the prosperity of one country.

The features in this dataset are:

* 0 country
* 1 incomeperperson
* 2 alcconsumption
* 3 armedforcesrate
* 4 breastcancerper100th
* 5 co2emissions
* 6 femaleemployrate
* 7 hivrate
* 8 internetuserate
* 9 lifeexpectancy
* 10 oilperperson
* 11 polityscore
* 12 relectricperperson
* 13 suicideper100th
* 14 employrate
* 15 urbanrate
* 16 alcconsumption5 **This column has been added to do tests**
* 17 suicideDanger **This colum has been aded to do the final report**

As a Software Engineer, I do not have a specific formation in this field. My report is only a test for the Data Analysis group
and is a fast approach of this real problem. Should be not taken as a research paper or report.

## Code:
The code developed is used for detecting patterns without having specific knowledge of the problem. It could be used to get a first approach to the main problem.

The code developed is a simple application console which uses sklearn and pandas tools to read an create the classification 
**Lasso regression**. We have added a simple tools to generate the graphics as we have learned in the **Week 3**. We have used the
created in **Week two** to obtain more results and compare it with this week.

> You can try the code at your own!.

### Implementation

Import needed libs
```python
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

pd.options.mode.chained_assignment = None
```

Read corpus
```python
corpus = pd.read_csv('corpus.csv')
```

Select, clean and normalice the features
```python
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
```

Model creation and test training
```python
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
```

The source could could be found [here](https://github.com/Duxy1996/ML-modules/blob/master/W4/main.py) and executed when you download all the dependencies.

### Usage demo

![GIF](../assets/Clustering.gif)

## Problem analysis

The code allows us to configure many parameters of the **clustering** classifier used to
know if a group of people is under the risk path. Using the features of several zones and countries in the world
we can know which zones and which groups of people are in danger.

In this week we are focus on knowing which features are the most important and we will compare the results with the 
given ones got in the **Weeks 2 and 3**.

## Clusters

This figure shows how we increase the number of claster the distance among them is reduced

![PIC](../assets/Figure_4_CV_100_20.png)

This figure shows

![PIC](../assets/Figure_4b_CV_100_20.png)

[2.0914294534689013, 1.7485648286851512, 1.4797656959058048,
1.321175351118435, 1.1719119977899328, 1.0924806129591975,
1.0337858553986206, 0.9783022694877036, 0.9369866496953962]

[1, 1.1960834503582978, 1.1816497932902865,
1.1200373172668683,1.1273673736679823, 1.0727073633055875,
1.0567765144532226, 1.0567141543481968, 1.0440941392342555] 

## Conclusions:

As we can see in the several iterations above that **alcohol consumption** , **employment rate**, **money income** and **urbanrate**
have a direct impact in the suicide rate in one country/zone. This factos are related with the people happines and it seems
normal the results given.

### Week one
* When there is a high level of alcohol consumption the risk of suicide is high too.
* When the employment rate is high, the suicide risk is low.
* When the employment rate is high, but the alocohol consumption too the suicide risk is high
* When the income is high, the suicide risk drops

### Week two
* The emplyment rate is not such important
* The urban rate is more important than **employment rate** and **money income**
* We cannot discard any features which affect to this risk despite of **oilperperson**

### Week three


## References:

[Zip python doc](https://docs.python.org/3.3/library/functions.html#zip)

[MathPlotlib python doc](https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py)

[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)

[Lasso Regression features](https://github.com/scikit-learn/scikit-learn/issues/6251)

[American Psychological Association](https://www.apa.org/monitor/2019/03/trends-suicide)

[Live Science web](https://www.livescience.com/62781-why-are-suicide-rates-rising.html)

[Desition trees](http://www.learnbymarketing.com/481/decision-tree-flavors-gini-info-gain/)