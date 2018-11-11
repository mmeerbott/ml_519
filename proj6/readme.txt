./main.py kmeans   datasets/iris_shuffled.csv >> output.txt
./main.py hsklearn datasets/iris_shuffled.csv >> output.txt
./main.py hscipy   datasets/iris_shuffled.csv >> output.txt
./main.py dbscan   datasets/iris_shuffled.csv >> output.txt

./main.py kmeans   datasets/faults.csv >> output.txt
./main.py hsklearn datasets/faults.csv >> output.txt
./main.py hscipy   datasets/faults.csv >> output.txt
./main.py dbscan   datasets/faults.csv >> output.txt

Note: iris data shuffled, values mapped: [virginica=>0, setosa=>1, versicolor=>2]
