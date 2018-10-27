Project 3 Report
================


Results
-----------
```
perceptron digits
Training Size (instances): 1437
Testing Size (instances): 360
Train Time (s): 0.032584129978204146
Test Time (s): 0.00030400799005292356
Accuracy: 0.8861111111111111

linsvm digits
Training Size (instances): 1437
Testing Size (instances): 360
Train Time (s): 0.03801236397703178
Test Time (s): 0.01098349000676535
Accuracy: 0.9305555555555556

nonlinsvm digits
Training Size (instances): 1437
Testing Size (instances): 360
Train Time (s): 0.36848922801436856
Test Time (s): 0.04052759599289857
Accuracy: 0.95

dtree digits
Training Size (instances): 1437
Testing Size (instances): 360
Train Time (s): 0.008211869979277253
Test Time (s): 0.0001666200114414096
Accuracy: 0.55

knn digits
Training Size (instances): 1437
Testing Size (instances): 360
Train Time (s): 0.0030904689920134842
Test Time (s): 0.05778124197968282
Accuracy: 0.9638888888888889

logreg digits
Training Size (instances): 1437
Testing Size (instances): 360
Train Time (s): 0.2595817540131975
Test Time (s): 0.0003204119857400656
Accuracy: 0.8972222222222223

perceptron realdisp
Training Size (instances): 143229
Testing Size (instances): 35808
Train Time (s): 8.038181312003871
Test Time (s): 0.01086566099547781
Accuracy: 1.0

linsvm, nonlinsvm, dtree, knn, logreg -- realdisp
<< STOPPED 30 min in >>
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

