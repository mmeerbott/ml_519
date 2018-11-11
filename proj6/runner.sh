#!/usr/bin/env bash

# requires you to comment out the plotting part

echo 'overwriting output.txt...'
rm -f output.txt

./main.py lda  datasets/iris_shuffled.csv >> output.txt
./main.py pca  datasets/iris_shuffled.csv >> output.txt
./main.py kpca datasets/iris_shuffled.csv >> output.txt

echo '-----' >> output.txt

./main.py lda  mnist >> output.txt
./main.py pca  mnist >> output.txt
./main.py kpca mnist >> output.txt

