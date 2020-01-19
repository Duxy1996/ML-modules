# Suicide risk prevention
### Exercise 1 from Machine Learning for Data Analysis

## Problem:
The suicide rate is raising in many countries which the problem seemed to be controlled.

## Dataset:
GapMinder is a good reference to get data from several countries and make a study to detect people which is in the suicide risk group and help with its problems before one fatality could appear.

As Software Engineer, I do not have specific formation in this field.

## Code:
The code developed is used for detecting patterns without having specific knowledge of the problem. It could be used to get a first approach with the main problem.

The code developed is a simple application console which uses sklearn and pandas tools to read an create the classification 
tree. We only have added a simple for loop which is used to selec the features that you want to evaluate and the imput of the
deepth of the tree.

The application show the test and train scores and generate an image with the binary tree generated.

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

```
corpus = pd.read_csv('corpus-2-2.csv')
```



## Problem analysis

The code allow us to configure at the moment many parameters of the **decision tree** classifier used to
know if a group of people is under the risk path. Using the features of several zones and contries in the world
we can know which zones and which groups of people are in danger.

The first approach made was to select the **alcohol consumption rate** to know if there is relation between  this
feature and the suicide probably.

* With 1-Depth decision tree configuration the score test is 85% and train 89%

![Alcohol-1-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alcohol-1-D.png)

* With 2-Depth decision tree configuration the score test is 77% and train 95%

![Alcohol-2-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alcohol-2-D.png)

The tree seems overfitted, the results are fine for the train group, but not for the test one

The second step was adding more features to the classification model (Decision tree):

Added **employ rate** feature

The employ rate usually is a good measure to know the happiness of one area and city.

* With 2-Depth decision tree configuration the score test is 88% and train 87%
![Alcohol-emply-2-D](https://raw.githubusercontent.com/Duxy1996/ML-modules/master/assets/Alco-emply-2-D.png)

* With 5-Depth decision tree configuration the score test is 76% and train 93%
![Alcohol-5-D]()

* With 7-Depth decision tree configuration the score test is 71% and train 97%
![Alcohol-7-D]()

We cannot find any other feature which improves the score given above but we can use other
features instead of the **alcohol consumption** and **employ rate** to know if they have an
impact in the area


## Conclusions:

## References:

[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)



