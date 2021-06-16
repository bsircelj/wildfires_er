from eventregistry import *
import json
from bs4 import BeautifulSoup
import nltk

# nltk.download('stopwords')
# from multiprocessing import Pool
# from sklearn.feature_extraction.text import CountVectorizer
# import os
# import time
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

lemmatizer = WordNetLemmatizer()


def read_file(path):
    with open(path) as file:
        lines = []
        for line in file.readlines():
            if len(line) > 3 and line[0:3] == 'eng':
                lines.append(line[0:-1])
    return lines


def article_to_words(text, language, lemmatize=False):
    # text = BeautifulSoup(raw, features="html5lib").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    stops = set(stopwords.words(language))
    meaningful_words = [w for w in words if not w in stops]
    if lemmatize:
        lemmatized_words = [lemmatizer.lemmatize(word) for word in meaningful_words]
        return " ".join(lemmatized_words)
    else:
        return " ".join(meaningful_words)


def retrieve_articles(er, event_uris):
    event_counter = 0
    # article_counter = 0
    articles = []
    # events = []
    for event_uri in event_uris:
        it = QueryEventArticlesIter(event_uri)
        print(f"Processing event {event_counter} of {len(event_uris)} total")
        # article_numbers = []
        for art in it.execQuery(er):
            if len(art) > 10:
                articles = np.concatenate((articles, [article_to_words(art['body'], "english")]))
                # article_numbers.append(article_counter)
                # article_counter += 1
        # events = np.concatenate((events, [article_numbers]))
        # events.append(article_numbers)
        event_counter += 1
    return articles


def retrieve_event_dict(er, event_uris):
    event_counter = 0
    # article_counter = 0
    event_dict = dict()
    # events = []
    for event_uri in event_uris:
        it = QueryEventArticlesIter(event_uri)
        print(f"Processing event {event_counter} of {len(event_uris)} total")
        # article_numbers = []
        articles = []
        for art in it.execQuery(er):
            if len(art) > 10:
                articles = np.concatenate((articles, [art]))
            else:
                print(event_uri)
                # article_numbers.append(article_counter)
                # article_counter += 1
        # events = np.concatenate((events, [article_numbers]))
        # events.append(article_numbers)
        event_dict[event_uri] = articles
        event_counter += 1
    return event_dict


if __name__ == '__main__':
    API_KEY = "3442278d-7990-4e69-a014-b7d029212520"
    er = EventRegistry(apiKey=API_KEY)
    true_dict = retrieve_event_dict(er, read_file('./wildfires/true_wildfire_events.txt'))
    false_dict = retrieve_event_dict(er, read_file('./wildfires/false_wildfire_events.txt'))

    # with open("wildfires/true_cleaned2.txt", "w") as file:
    #     for event in true_dict.keys():
    #         file.write(f'{event}\n')
    #
    # with open("wildfires/false_cleaned2.txt", "w") as file:
    #     for event in false_dict.keys():
    #         file.write(f'{event}\n')

    with open(f"wildfires/true.dict", 'wb') as file:
        pickle.dump(true_dict, file)

    with open(f"wildfires/false.dict", 'wb') as file:
        pickle.dump(false_dict, file)
