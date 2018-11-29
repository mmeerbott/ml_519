#!/usr/bin/env bash

# requires you to comment out the plotting part

echo 'overwriting output.txt...'
rm -f output.txt

./main.py bagging    digits >> output.txt
./main.py randforest digits >> output.txt
./main.py adaboost   digits >> output.txt

echo '-----' >> output.txt

./main.py bagging    datasets/mammographic_masses.data.txt >> output.txt
./main.py randforest datasets/mammographic_masses.data.txt >> output.txt
./main.py adaboost   datasets/mammographic_masses.data.txt >> output.txt

