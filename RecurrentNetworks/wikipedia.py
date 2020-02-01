# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:55:02 2020

@author: Guru Prasad Muppana
Wiki Pedia Reading.
"""

import numpy as np
import os
import string



def get_wikipedia_data():
    pass

def remove_punctuation(line):
    return line.translate(str.maketrans('','',string.punctuation))
    

def get_robert_frost():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    
    for line in open("D:/python/Guru/Datasets/RobertFrost/robert_frost.txt"):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx


if __name__ == "__main__" :
    sentences, word2idx = get_robert_frost()
    
    