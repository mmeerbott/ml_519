Project 3 Report
================


Results
-----------
```
linreg datasets/housing.data.txt
Training Size (instances): 404
Testing Size (instances): 102
Train Time (s): 0.0010534149478189647
Test Time (s): 0.0006580009940080345
Data slope: -0.070, intercept: 34.319
MSE train: 42.585, test: 22.278
R^2 train: 0.502, test: 0.171


ransac datasets/housing.data.txt
Training Size (instances): 404
Testing Size (instances): 102
Train Time (s): 0.011309213994536549
Test Time (s): 0.0007798140286467969
Data slope: -0.070, intercept: 34.319
MSE train: 73.027, test: 118.721
R^2 train: 0.146, test: -3.415


ridge datasets/housing.data.txt
Training Size (instances): 404
Testing Size (instances): 102
Train Time (s): 0.001593180000782013
Test Time (s): 0.0010018310276791453
Data slope: -0.070, intercept: 34.319
MSE train: 42.585, test: 22.277
R^2 train: 0.502, test: 0.171


lasso datasets/housing.data.txt
Training Size (instances): 404
Testing Size (instances): 102
Train Time (s): 0.0009100159513764083
Test Time (s): 0.0006537479930557311
Data slope: -0.070, intercept: 34.319
MSE train: 42.619, test: 22.998
R^2 train: 0.502, test: 0.145


nonlinear datasets/housing.data.txt
Training Size (instances): 404
Testing Size (instances): 102
Train Time (s): 0.0007939990027807653
Test Time (s): 0.0006739479722455144
MSE train: 22.694, test: 28.444
R^2 train: 0.735, test: -0.058

linreg datasets/all_breakdown.csv 
Training Size (instances): 54067 
Testing Size (instances): 13517 
Train Time (s): 0.003546730033122003 
Test Time (s): 0.007904438010882586 
Data slope: 6.701, intercept: 141.999 
MSE train: 1058246.451, test: 1012001.116 
R^2 train: 0.025, test: 0.028 
 
 
ransac datasets/all_breakdown.csv 
Training Size (instances): 54067 
Testing Size (instances): 13517 
Train Time (s): 0.15652674197917804 
Test Time (s): 0.011446275981143117 
Data slope: 6.701, intercept: 141.999 
MSE train: 1237500.466, test: 1152224.602 
R^2 train: -0.140, test: -0.107 
 
 
ridge datasets/all_breakdown.csv 
Training Size (instances): 54067 
Testing Size (instances): 13517 
Train Time (s): 0.001984770002309233 
Test Time (s): 0.0016921749920584261 
Data slope: 6.701, intercept: 141.999 
MSE train: 1058246.451, test: 1012001.116 
R^2 train: 0.025, test: 0.028 
 
 
lasso datasets/all_breakdown.csv 
Training Size (instances): 54067 
Testing Size (instances): 13517 
Train Time (s): 0.0030993890250101686 
Test Time (s): 0.0021569780074059963 
Data slope: 6.701, intercept: 141.999 
MSE train: 1058246.453, test: 1012000.774 
R^2 train: 0.025, test: 0.028 
 
 
nonlinear datasets/all_breakdown.csv 
Training Size (instances): 54067 
Testing Size (instances): 13517 
Train Time (s): 0.012229336949530989 
Test Time (s): 0.0023268309887498617 
MSE train: 984166.807, test: 958451.689 
R^2 train: 0.093, test: 0.079
```

Analysis
-------------
The classifiers react heavily on their parameters. I mostly used the ones
used in the notes, but when I changed some of them, the runtime increased and
they all took a long time, so I wouldn't let them finish (Ctrl-C). Notably,
when I changed C to a lower number in SVC, the runtime jumped and was running
for tens of minutes. I couldn't get all of the runtimes for the REALDISP
dataset. In retrospect, I think I should have cut down the number of instances.

I used `subject1_ideal.log` from 
[REALDISP](https://archive.ics.uci.edu/ml/datasets/REALDISP+Activity+Recognition+Dataset)

Decision Tree Classifier
------------------------
The documentation states that the Decision Tree uses `max depth` and `min sample
leaves`. By default, these are set to allow the tree to grow unpruned and can
take up a lot of memory. This info can be found at the Notes section of the
documentation.

The source code when they're implemented can be found [here](https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/tree/tree.py#L75)

`min_samples_leaf` can be seen on line 175 and 205. It is used to calculate
`min_samples_split`, by being multiplied as a factor to create the minimum
number of decision splits. `min_samples_leaf` can be seen again in another
[file](https://github.com/scikit-learn/scikit-learn/blob/a7e17117bb15eb3f51ebccc1bd53e42fcb4e6cd8/sklearn/tree/_tree.pyx#L455).


`max depth` can also be found in the same files, on line 534 
[here](https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/tree/tree.py#L534),
and line 453
[here](https://github.com/scikit-learn/scikit-learn/blob/a7e17117bb15eb3f51ebccc1bd53e42fcb4e6cd8/sklearn/tree/_tree.pyx#L455).
It controls the maximum depth the tree will grow to.

