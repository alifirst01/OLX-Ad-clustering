import re
import os
import nltk
import math
import numpy as np
from nltk import edit_distance
from collections import Counter
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import euclidean_distances


stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def counter_manhattan_distance(c1, c2):
    vecA = []
    vecB = []
    terms = set(c1).union(c2)
    for k in terms:
        vecA.append(c1.get(k, 0))
        vecB.append(c2.get(k, 0))
    return float(manhattan_distance(vecA, vecB))

def counter_euclidean_similarity(c1, c2):
    vecA = []
    vecB = []
    terms = set(c1).union(c2)
    for k in terms:
        vecA.append(c1.get(k, 0))
        vecB.append(c2.get(k, 0))
    return float(euclidean_distances(vecA, vecB, squared=True))

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    if magA == 0 or magB == 0:
        return 0
    else:
        return dotprod / (magA * magB)

def counter_similarity(desc1, desc2, c = 0):
    listA = tokenize_and_stem(desc1)
    listB = tokenize_and_stem(desc2)
    counterA = Counter(listA)
    counterB = Counter(listB)
    if c == 0:
        return counter_cosine_similarity(counterA, counterB)
    elif c == 1:
        return counter_euclidean_similarity(counterA, counterB)
    else:
        return counter_manhattan_distance(counterA, counterB)


def edit_distance_similarity(desc1, desc2):
    listA = desc1.split(' ')
    listB = desc2.split(' ')
    scores = []
    for w1 in listA:
        for w2 in listB:
            scores.append(edit_distance(w1, w2))
    return np.mean(scores)


def show_images(images, cluster, DIR, cols=1, titles=None):
    plt.close('all')
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.axis('off')
        plt.imshow(image)
        #a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.savefig(os.path.join('Results', DIR, str(cluster) + '.png'))
    #plt.show()
