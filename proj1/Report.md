Project 1 Report
================


Results
-----------
### Perceptron
Accuracy: 60.4%

### Adaline
Accuracy: 39.6%

### Stochastic Gradient Descent
Accuracy: 60.4%

### SGD - One Versus Rest
Accuracy: 1.2%

Analysis
-------------
It seems that every prediction yielded an array of 1s or -1s, meaning something went
wrong in the code/logic. I am not sure what it is at this point (if I did, it'd be fixed
by now), but my best guess right now is that np.where is having trouble comparing things
with strings, though the fact that y was separated correctly in the first 3 algorithms
says differently. Submitting at this point since I'm already late and won't have time to
work on it tomorrow.
