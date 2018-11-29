./main.py lda  datasets/iris_shuffled.csv >> output.txt
./main.py pca  datasets/iris_shuffled.csv >> output.txt
./main.py kpca datasets/iris_shuffled.csv >> output.txt

./main.py lda  mnist >> output.txt
./main.py pca  mnist >> output.txt
./main.py kpca mnist >> output.txt

Note: iris data shuffled, values mapped:[virginica=>0, setosa=>1, versicolor=>2]

