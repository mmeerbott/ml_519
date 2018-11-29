./main.py bagging    digits >> output.txt
./main.py randforest digits >> output.txt
./main.py adaboost   digits >> output.txt

./main.py lda  datasets/mammographic_masses.data.txt >> output.txt
./main.py pca  datasets/mammographic_masses.data.txt >> output.txt
./main.py kpca datasets/mammographic_masses.data.txt >> output.txt


Note: removed all '?'s in the dataset
