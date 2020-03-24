import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def makeCSVFile(amountOfReviews) :
    # amountOfReviews = 1
    createCSVS(amountOfReviews)

    negativeData = pd.read_csv('negativeReviews.csv')
    positiveData = pd.read_csv('positiveReviews.csv')

    extendedData = negativeData.merge(positiveData, how='outer')
    del(extendedData[list(extendedData)[0]])
    # print(extendedData)
    extendedData.to_csv('data.csv')

def createCSVS(amountOfReviews) :
    functions = [searchForNegativeOverviews, searchForPositiveOverviews]
    procs = []

    for func in functions :
        p = mp.Process(target=func, args=(amountOfReviews,))
        procs.append(p)
        p.start()

    for p in procs :
        p.join()

def searchForNegativeOverviews(amountOfReviews) :
    listOfNegativeFiles = os.listdir(path='/home/maxwell/Big Data/aclImdb_v1/aclImdb/train/neg')
    listOfReviews = []

    for i in range(amountOfReviews) :
        # print(listOfNegativeFiles[i])
        file = open(f'/home/maxwell/Big Data/aclImdb_v1/aclImdb/train/neg/{listOfNegativeFiles[i]}')
        # file = open(listOfNegativeFiles[i])
        content = file.read()

        listOfReviews.append({'Review' : content, 'Assessment' : 0})
        file.close()

    dataFrame = pd.DataFrame(listOfReviews)
    dataFrame.to_csv('negativeReviews.csv')

def searchForPositiveOverviews(amountOfReviews) :
    listOfPositiveFiles = os.listdir(path='/home/maxwell/Big Data/aclImdb_v1/aclImdb/train/pos')
    listOfReviews = []

    for i in range(amountOfReviews) :
        # print(listOfPositiveFiles[i])
        file = open(f'/home/maxwell/Big Data/aclImdb_v1/aclImdb/train/pos/{listOfPositiveFiles[i]}')
        # file = open(listOfPositiveFiles[i])
        content = file.read()

        listOfReviews.append({'Review' : content, 'Assessment' : 1})
        file.close()

    dataFrame = pd.DataFrame(listOfReviews)
    dataFrame.to_csv('positiveReviews.csv')

def getWordsRate(table, label) :
    count = CountVectorizer(max_features=5)
    count.fit_transform(table['Review'])

    tfidf = TfidfVectorizer().fit(table['Review'])
    featureArr = np.array(tfidf.get_feature_names())

    n = 5
    print(f'Top 5 words used in {label} reviews:   {list(count.vocabulary_)}')
    print(f'Top 5 meaningful words used in {label} reviews:   {featureArr[:n]}')

if __name__ == "__main__" :
    amountOfReviews = int(input('Enter number of reviews:   '))
    makeCSVFile(amountOfReviews)

    count = CountVectorizer()
    tfidf = TfidfTransformer()

    table = pd.read_csv('data.csv',)

    bag = count.fit_transform(table['Review'])
    bag_tfidf = tfidf.fit_transform(bag)

    # vocabulary = {k: count.vocabulary_[k] for k in sorted(count.vocabulary_, key=count.vocabulary_.get, reverse=False)}
    # print(vocabulary)
    print('Bag of words:    ',bag.toarray(),sep='\n')
    np.set_printoptions(precision=3)
    print('Words Tf-Idf:    ',bag_tfidf.toarray(),sep='\n')

    negativeTable = table.head(amountOfReviews)
    positiveTable = table.tail(amountOfReviews)

    getWordsRate(negativeTable,'negative')
    getWordsRate(positiveTable,'positive')
