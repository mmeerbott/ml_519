#!/usr/bin/env bash

./main.py perceptron digits >> output.txt
./main.py linsvm     digits >> output.txt
./main.py nonlinsvm  digits >> output.txt
./main.py dtree      digits >> output.txt
./main.py knn        digits >> output.txt
./main.py logreg     digits >> output.txt

./main.py perceptron realdisp >> output.txt
./main.py linsvm     realdisp >> output.txt
./main.py nonlinsvm  realdisp >> output.txt
./main.py dtree      realdisp >> output.txt
./main.py knn        realdisp >> output.txt
./main.py logreg     realdisp >> output.txt
