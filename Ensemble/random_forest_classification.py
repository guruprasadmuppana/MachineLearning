# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:24:10 2020

@author: Guru Prasad Muppana

Random forest.



"""

import numpy as np
import pandas as pd

# Data files can be found at:
# https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/

NUMERICAL_COLS = ()
CATEGORICAL_COLS = np.arange(22) + 1 # starts with 1 and goes until 22 inclusive.


def get_mushroom_data():
    df = pd.read_csv("D:\python\Guru\Datasets\Mushroom\mushroom.data")
    print("file opening was successful")

def main():
    print("main function starts here !")
    get_mushroom_data()

if __name__ == "__main__" :
    main()
