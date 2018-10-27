#!/usr/bin/env bash

# requires you to comment out the plotting part

./main.py linreg     datasets/housing.data.txt >> output.txt
./main.py ransac     datasets/housing.data.txt >> output.txt
./main.py ridge      datasets/housing.data.txt >> output.txt
./main.py lasso      datasets/housing.data.txt >> output.txt
./main.py nonlinear  datasets/housing.data.txt >> output.txt

./main.py linreg     datasets/all_breakdown.csv >> output.txt
./main.py ransac     datasets/all_breakdown.csv >> output.txt
./main.py ridge      datasets/all_breakdown.csv >> output.txt
./main.py lasso      datasets/all_breakdown.csv >> output.txt
./main.py nonlinear  datasets/all_breakdown.csv >> output.txt

