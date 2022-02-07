# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:51:32 2022

@author: ehsan
"""

#from PyDictionary import PyDictionary
#dictionary=PyDictionary()

#words = dictionary.

#joblib parallelisation 

import nltk
import random 
nltk.download('wordnet')
from nltk.corpus import wordnet 

#syns = wordnet.synsets("pup")
#sims = []
#miss_word = 'desing'
#for word in wordnet:
#    sims.append([word.wup_similarity(miss_word), word])
    
#sims_sorted = sorted(sims, key=itemgetter(1))
#array.sort(key = lambda x:x[1]) 
print(len(list(wordnet.words())))

list_correct = ["ABILITY","ABROAD","ACADEMIC","ACCESSION","ACCOMMODATE","ACCORDANCE","ACCURATELY","ACHIEVED","ACHIEVED","ADDITIONAL"]
list_wrong = ["ABILTY","ABRAOD","ACEDEMIC","ACCESION","ACCOMODATE","ACORDANCE","ACCRUATELY","ACIEVED","ACCHEIVED","ADDITONAL"]

#change k = 10, 100, 1000 
list_whole = random.choices(list(wordnet.words()), k=10)
#select 1000 words with starting letter as "a"

import numpy

def levenshtein(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))
    for t1 in range(len(token1) + 1): distances[t1][0] = t1
    for t2 in range(len(token2) + 1): distances[0][t2] = t2
    a = 0; b = 0; c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
              a = distances[t1][t2 - 1]
              b = distances[t1 - 1][t2]
              c = distances[t1 - 1][t2 - 1]

              if (a <= b and a <= c): distances[t1][t2] = a + 1
              elif (b <= a and b <= c): distances[t1][t2] = b + 1
              else: 
                if (token1[t1 - 1] == token2[t2 - 1]): distances[t1][t2] = c
                else: distances[t1][t2] = c + 2
                
    return distances[len(token1)][len(token2)]

matrix1 = [[0 for i in range(10)] for j in range(10)]
matrix2 = [[0 for i in range(10)] for j in range(10)]

matrix_big = [[[0 for i in range(10)] for j in range(2)] for k in range(10)]

for i in range(0,10):
    for j in range(0,10):
        matrix1[i][j] = levenshtein(list_whole[i],list_wrong[j])
        
for i in range(0,10):
    for j in range(0,10):
        matrix2[i][j] = levenshtein(list_correct[i],list_wrong[j])
        
#replace with 1000 and 10 respectively 
#for i in range(0,2):
#    for j in range(0,10):
#        matrix1[i][j] = levenshtein(list_whole[i],list_wrong[j])
   
#checking with correct word's list 
#replacing list_whole with list_correct 
#for i in range(0,2):
#    for j in range(0,10):
#        matrix1[i][j] = levenshtein(list_whole[i],list_wrong[j])
        

print("matrix of MEDs", matrix1)
print()
print("Wrong words", list_wrong)
print()
print("Candidate words",list_whole)
print()

#import pytrec_eval

#for miss, correct in word_pairs:
#    list1 = dictionary.get_similar(miss)
#    list_whole.append(list1.append([miss,correct]))

import numpy as np

count = []
#print(count)

def s_and_k(matrix, k=[1, 5, 10]):
    matrix = np.array(matrix)
    for i in range(matrix.shape[0]):
        #print(matrix[i])
        sorted1 = np.argsort(matrix[i])
        #replacing list_whole with list_correct
        sorted_words_candidate = np.take(list_whole, sorted1)
        #print(sorted_words_candidate)
        correct_word = list_correct[i]
        for j in k:
            if correct_word in sorted_words_candidate[:j]:
                #print(1)
                count.append("1")
                #print(correct_word)
            else:
                #print("0")
                count.append("0")
        print("count list for k =", k, "for word index at ", i, count)
        count.clear()

count1 = []

def s_and_k_correct(matrix, k=[1, 5, 10]):
    matrix = np.array(matrix)
    for i in range(matrix.shape[0]):
        #print(matrix[i])
        sorted1 = np.argsort(matrix[i])
        #replacing list_whole with list_correct
        sorted_words_correct = np.take(list_correct, sorted1)
        correct_word = list_correct[i]
        for j in k:
            if correct_word in sorted_words_correct[:j]:
                #print(1)
                count1.append("1")
                #print(correct_word)
            else:
                #print("0")
                count1.append("0")
        print("count list for k =", k, "for word index at ", i, count1)
        count1.clear()

            

'''
counters = []

def s_and_k(matrix,k=10):
    for i in range(1):
        #if word at 
        counter = 0
        for j in range(k):
            #select index with min med value 
            index_min = np.argmin(np.array(matrix)[:,i])
            print(index_min)
            print()
            #find the corresponding word in the list_whole using the given index 
            word_at_min_index = list_whole[index_min]
            #remove word_at_min_index from the list_whole 
            print(word_at_min_index)
            print()
            #check if the word is in the or is the word of list_correct
            if(word_at_min_index == list_correct[0]):
                return 1 
    return 0
'''
print("For candidate words", s_and_k(matrix1,k=[1,5,10]))

print("For correct words", s_and_k_correct(matrix2,k=[1,5,10]))
