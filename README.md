# Suicide risk prevention
### Exercise 1 from Machine Learning for Data Analysis

## Problem:
The suicide rate is raising in many countries which the problem seemed to be controlled. The main factors of this case are not
well known yet, and this fact is dangerous because we cannot do anything without knowing the problem.

In countries such us United Kindom and EEUU suicides are rising and, now the average is over the 15 people over 100.000 people which
is worrying.

We want to use decision trees to predict in which zones this rate can raise. Using the **GapMinder** dataset we will discover which
zones are over the 15 per 100.000 of suicide rate. Knowing this information wee could make a first approach to the problem and help the
people which is in the risk group.

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
tree. We only have added a simple for loop which is used to select the features that you want to evaluate and the impact of the
depth of the tree.

The application show the test and train scores and generate an image with the binary tree generated.

### Implementation
Import needed libs
```python
import pandas as pd
import numpy as np
import pydot

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```

Read corpus

```python
corpus = pd.read_csv('corpus.csv')
```

Feature selection

```python
# Show all the features loaded from the corpus
for i,col in enumerate(corpus.columns): 
  print(i,col)
  dic[i] = col

features = []
feature_sel = 0

# Allows you to selec the features using the enumeration number
while(feature_sel != -1):
  feature_sel = int(input("> "))
  if (feature_sel >= 0):
    features.append(dic[feature_sel])
    print(features)
```

Clasificator init

```python
# Split the datasaet in two and train the model
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_trian, random_state = 0)

clas = DecisionTreeClassifier(max_depth = depth).fit(X_train, y_train)
pred = clas.predict(X_test)
```


### Usage demo

![Alcohol-1-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/DemoGif.gif)

## Problem analysis

The code allows us to configure at the moment many parameters of the **decision tree** classifier used to
know if a group of people is under the risk path. Using the features of several zones and countries in the world
we can know which zones and which groups of people are in danger.

The first approach made was to select the **alcohol consumption rate** to know if there is relation between  this
feature and the suicide probably. This feature is added as X[0] and its measured in the quantity of peoplw which has
problems with alcohol.

### Representation

Using one desition tree create with the console application we are going to explain which means each symbol:

![Alcohol-1-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alcohol-1-D.png)

` X[0] < 14.93 ` -> If the alcohol consumption is under this quantity the split is done in the left node, if not to the right
one.

`gini = 0.289` -> Gini Index is calculated by subtracting the sum of the squared probabilities of each class from one
`samples = 80` -> All the samples which are in this node.
`value = [66,14]` -> All the samples clasified which are in this node. When one of them is 0 the classiffication in perfect and the
gini index is 0

* With 1-Depth decision tree configuration the score test is 85% and train 89%

![Alcohol-1-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alcohol-1-D.png)

* With 2-Depth decision tree configuration the score test is 77% and train 95%

![Alcohol-2-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alcohol-2-D.png)

The tree seems overfitted, the results are fine for the train group, but not for the test one

The second step was adding more features to the classification model (Decision tree):

Added **employ rate** feature

The employ rate usually is a good measure to know the happiness of one area and city. This measure is represented with the
X[1] variable and represents the emplyment rate in one zone.

* With 2-Depth decision tree configuration the score test is 88% and train 87%

![Alcohol-emply-2-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alco-emply-2-D.png)

* With 5-Depth decision tree configuration the score test is 76% and train 93%

![Alcohol-5-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alco-emply-5-D.png)

* With 7-Depth decision tree configuration the score test is 71% and train 97%

![Alcohol-7-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alco-emply-7-D.png)

Increasing the depth path is overfitting the model instead of improving it. The data generalization in the level 2 of
depth is enough to classify well the test results.

Added **avarage income** feature

The income is usually a good reference to identify exclussion patterns and money problems which usually is related with
suicide problems. This feature is represented with the X[2] variable. Is the amount of money recieved.

* With 2-Depth decision tree configuration the score test is 90% and train 87% **(Best score found)**

![Alcohol-7-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/income-alco-emply-2-D.png)

* With 5-Depth decision tree configuration the score test is 87% and train 94%

![Alcohol-7-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/income-alco-emply-5-D.png)

* With 7-Depth decision tree configuration the score test is 82% and train 96%

![Alcohol-7-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/income-alco-emply-7-D.png)


## Conclusions:

## References:

[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
[American Psychological Association](https://www.apa.org/monitor/2019/03/trends-suicide)
[Live Science web](https://www.livescience.com/62781-why-are-suicide-rates-rising.html)



