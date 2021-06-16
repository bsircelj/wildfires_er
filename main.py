import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import re
from bs4 import BeautifulSoup
import nltk
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import os
import time
from nltk.corpus import stopwords
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from eventregistry import *
from retrieve_articles import retrieve_articles
import random

API_KEY = "3442278d-7990-4e69-a014-b7d029212520"


# nltk.download('wordnet')

def read_file(path):
    with open(path) as file:
        lines = []
        for line in file.readlines():
            if len(line) > 3 and line[0:3] == 'eng':
                lines.append(line[0:-1])
    return lines


#
#
# def classify_articles():
#     er = EventRegistry(apiKey=API_KEY)
#     # event_uris = ["eng-6528950", "eng-6522231", "eng-6522143", "deu-1315624", "eng-6520380", "eng-6511760",
#     #               "eng-6503104"]
#
#     # true_articles = retrieve_articles(er, read_file('./wildfires/true_wildfire_events.txt'))
#     # false_articles = retrieve_articles(er, read_file('./wildfires/false_wildfire_events.txt'))
#     # np.savetxt('true_articles_lemm.txt', true_articles, fmt='%s')
#     # np.savetxt('false_articles_lemm.txt', false_articles, fmt='%s')
#
#     true_articles = np.loadtxt(f'true_articles.txt', dtype='str', delimiter='\n')
#     false_articles = np.loadtxt(f'false_articles.txt', dtype='str', delimiter='\n')
#
#     # true_articles = retrieve_articles(er, ['eng-6462304', 'eng-6449317'])
#     # false_articles = retrieve_articles(er, ['eng-6133953', 'eng-6127892'])
#
#     # for max_features in [1000, 1500, 2000, 3000, 4000, 6000, 999999]:
#     for max_features in [4000, 5000, 6000, None]:
#         # vectorizer = CountVectorizer(analyzer="word",
#         #                              max_features=max_features
#         #                              )
#         vectorizer = TfidfVectorizer()
#
#         true_len = len(true_articles)
#         false_len = len(false_articles)
#
#         # print(f'True articles: {true_len}, false articles: {false_len}')
#
#         y = np.concatenate((np.ones(true_len, np.uint8), np.zeros(false_len, np.uint8)))
#
#         x = vectorizer.fit_transform(np.concatenate((true_articles, false_articles)))
#         vocabulary = vectorizer.vocabulary_
#
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
#
#         model = SVC(gamma='scale',
#                     cache_size=12000,
#                     class_weight='balanced',
#                     max_iter=-1)
#         model.fit(x_train, y_train)
#
#         y_pred = model.predict(x_test)
#
#         precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
#         print(f'Max features: {max_features}')
#         print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')


def create_event_model(true_id, false_id):
    # event_uris = ["eng-6528950", "eng-6522231", "eng-6522143", "deu-1315624", "eng-6520380", "eng-6511760",
    #               "eng-6503104"]
    er = EventRegistry(apiKey=API_KEY)
    folder = 'wildfire_model'

    # true_articles = retrieve_events(er, read_file('./wildfires/true_wildfire_events.txt'))
    # false_articles = retrieve_events(er, read_file('./wildfires/false_wildfire_events.txt'))
    true_articles = retrieve_articles(er, true_id)
    false_articles = retrieve_articles(er, false_id)

    np.savetxt('true_events.txt', true_articles, fmt='%s')
    np.savetxt('false_events.txt', false_articles, fmt='%s')

    true_articles = np.loadtxt(f'true_events.txt', dtype='str', delimiter='\n')
    false_articles = np.loadtxt(f'false_events.txt', dtype='str', delimiter='\n')

    # true_uris = read_file('./wildfires/true_wildfire_events.txt')
    # false_uris = read_file('./wildfires/false_wildfire_events.txt')

    # for max_features in [1000, 1500, 2000, 3000, 4000, 6000, 999999]:
    vectorizer = TfidfVectorizer(use_idf=True)
    # vectorizer = CountVectorizer(analyzer="word", max_features=6000)

    true_len = len(true_articles)
    false_len = len(false_articles)

    # print(f'True articles: {true_len}, false articles: {false_len}')

    y = list(np.concatenate((np.ones(true_len, np.uint8), np.zeros(false_len, np.uint8))))
    joined_articles = np.concatenate((true_articles, false_articles))
    # print(joined_articles[0:10])

    x = vectorizer.fit_transform(joined_articles)
    with open(f"{folder}/wildfires.vectorizer", 'wb') as file:
        pickle.dump(vectorizer, file)

    # dense_x = x.todense()
    # print(dense_x[100:150, 5:10])
    # elements=10
    # linep = ""
    # for line in dense_x[0:elements]:
    #     for word in line[0:elements]:
    #         linep += str(word)
    #     linep += '\n'
    # print(linep)
    # vocabulary = list(vectorizer.vocabulary_.keys())
    # idf = vectorizer.idf_

    # np.savetxt(f'{folder}/vocabulary.txt', vocabulary, fmt='%s')
    # np.savetxt(f'{folder}/idf.txt', idf, fmt='%f')

    model = SVC(gamma='scale',
                cache_size=12000,
                class_weight='balanced',
                max_iter=-1)
    model.fit(x, y)

    with open(f"{folder}/wildfires.model", 'wb') as file:
        pickle.dump(model, file)


def classify_event(event_id, folder):
    er = EventRegistry(apiKey=API_KEY)
    articles = retrieve_articles(er, [event_id])
    print(f'Articles loaded: {len(articles)}')
    if len(articles) == 0:
        return 0

    # print(articles[0])
    # vocabulary = np.loadtxt(f'{folder}/vocabulary.txt', dtype='str', delimiter='\n')
    # idf = np.loadtxt(f'{folder}/idf.txt', dtype='float', delimiter='\n')
    # vectorizer = TfidfVectorizer(vocabulary=vocabulary, use_idf=True)
    # print(articles[0:10])
    # x = vectorizer.transform(articles)
    # dense_x = x.todense()
    # print(dense_x[100:150, 5:10])

    # vectorizer.idf_ = idf
    # print(articles[0:10])
    # x = vectorizer.transform(articles)
    # dense_x = x.todense()
    # print(dense_x[100:150, 5:10])
    # elements = 10
    # linep = ""
    # for line in dense_x[0:elements]:
    #     for word in line[0:elements]:
    #         linep += str(word)
    #     linep += '\n'
    # print(linep)

    with open(f"{folder}/wildfires.vectorizer", 'rb') as file:
        vectorizer = pickle.load(file)

    with open(f"{folder}/wildfires.model", 'rb') as file:
        model = pickle.load(file)

    x = vectorizer.transform(articles)
    y_pred = model.predict(x)

    print('individual articles')
    print(y_pred)

    if sum(y_pred) / len(y_pred) >= 0.5:
        print(1)
        return 1
    else:
        print(0)
        return 0


if __name__ == '__main__':

    with open(f"wildfires/true.dict", 'rb') as file:
        true_dict = pickle.load(file)

    with open(f"wildfires/false.dict", 'rb') as file:
        false_dict = pickle.load(file)

    true_events = read_file('./wildfires/true_wildfire_events.txt')
    false_events = read_file('./wildfires/false_wildfire_events.txt')
    split = 0.66
    true_train, true_test, false_train, false_test = [], [], [], []

    random.seed(534245)

    for ev in true_events:
        if random.random() < split:
            true_train.append(ev)
        else:
            true_test.append(ev)

    for ev in false_events:
        if random.random() < split:
            false_train.append(ev)
        else:
            false_test.append(ev)

    print('train test split')
    print(f'[Train] true: {len(true_train)} false: {len(false_train)}')
    print(f'[Test] true: {len(true_test)} false: {len(false_test)}')

    create_event_model(true_train, false_train)

    true_len = len(true_test)
    false_len = len(false_test)

    y_test = list(np.concatenate((np.ones(true_len), np.zeros(false_len))))
    print(y_test)
    y_pred = []

    # classify_event(np.concatenate((true_train, false_train)), 'wildfire_model')

    for event in true_test:
        # for event in true_train:
        print('\ntrue')
        print(event)
        y_pred.append(classify_event(event, 'wildfire_model'))
        # if y_pred[-1] == 0:
        #     print(false_events[event][0])

    for event in false_test:
        # for event in false_train:
        print('\nfalse')
        print(event)
        y_pred.append(classify_event(event, 'wildfire_model'))
        # if y_pred[-1] == 1:
        #     print(false_events[event][0])

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
    print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
    # classify_articles()
