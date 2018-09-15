#!/usr/bin/env python3.6

import argparse




if __name__=='__main__':
    # handle arguments
    parser = argparse.ArgumentParser(
                 description='Runs Perceptron/Adaline/MGD models on a dataset'
             )

    parser.add_argument('model', help='[Perceptron/Adaline/MGD]')
    parser.add_argument('dataset', help='[dataset.zip]')

    args = parser.parse_args()
    # access like args.model -- they'll go in order as well
