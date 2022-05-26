import os
import re
import time
import argparse
import string
from collections import defaultdict

import nltk
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling
        self.k = 5
        self.__w2i = {}
        self.__i2w = {}
        self.__V = 0
        self.__unigram_distribution = []
        self.__corrected_unigram_distribution = []
        self.__W = None
        self.__U = None

    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w

    @property
    def vocab_size(self):
        return self.__V

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        line = line.split()
        words = []
        for w in line:
            w = re.sub('[^A-Za-z]+', r"", w)
            if w != "":
                words.append(w)
        return words

    def cleaning(self, text):
        word_buffer = ''
        delimiter = {'\n', ' ', '!', '$', '&', "'", ',', '-', '.', ':', ';', '?'}
        filtered = []
        for i in text:
            if i in delimiter:
                if word_buffer != '':
                    filtered.append(word_buffer)
                if i == "'":
                    word_buffer = "'"
                else:
                    filtered.append(i)
                    word_buffer = ''
            else:
                word_buffer += i
        if word_buffer != '':
            filtered.append(word_buffer)
        return filtered

    def text_gen(self):
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    # yield self.clean_line(line)
                    # yield nltk.word_tokenize(line)
                    yield self.cleaning(line.lower()) + ["\n"]

    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        return [self.__w2i[sent[j]] for j in range(max(0, i - self.__lws), min(len(sent), i + self.__rws + 1)) if
                j != i]

    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        self.__w2i = {}
        self.__i2w = {}
        i = 0
        self.__unigram_distribution = []
        for line in self.text_gen():
            for w in line:
                if w not in self.__w2i.keys():
                    self.__w2i[w] = i
                    self.__i2w[i] = w
                    i += 1
                    self.__unigram_distribution.append(1)
                else:
                    self.__unigram_distribution[self.__w2i[w]] += 1

        self.__V = len(self.__unigram_distribution)
        self.__unigram_distribution = np.array(self.__unigram_distribution) / np.sum(self.__unigram_distribution)
        self.__corrected_unigram_distribution = self.__unigram_distribution ** 0.75
        self.__corrected_unigram_distribution /= np.sum(self.__corrected_unigram_distribution)

        focus_words = []
        context_words = []
        for line in self.text_gen():
            for i in range(len(line)):
                focus_words.append(self.__w2i[line[i]])
                context_words.append(self.get_context(line, i))
        return focus_words, context_words

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        if self.__use_corrected:
            probs = np.copy(self.__corrected_unigram_distribution)
        else:
            probs = np.copy(self.__unigram_distribution)
        probs[xb] = 0
        probs[pos] = 0
        probs = probs / sum(probs)
        return np.random.choice(range(self.__V), size=number, replace=True, p=probs)

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION
        self.__W = np.random.normal(0, 1, size=(self.__V, self.__H))
        self.__U = np.random.normal(0, 1, size=(self.__V, self.__H))

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):
                lr = self.__lr * max(0.0001, (1 - (N - i) / ((self.__epochs - ep) * N + 1)))
                focus = self.__W[x[i]]
                gradient_w = np.zeros(self.__H)
                temp_U = np.copy(self.__U[t[i]])
                for j in range(len(t[i])):
                    context = self.__U[t[i][j]]
                    negative_indices = self.negative_sampling(self.__nsample, x[i], t[i][j])
                    negatives = self.__U[negative_indices]
                    gradient_w += context * (self.sigmoid(context @ focus.T) - 1) + negatives.T @ self.sigmoid(
                        negatives @ focus.T)
                    gradient_u_pos = focus * (self.sigmoid(context @ focus.T) - 1)
                    gradient_u_neg = np.outer(self.sigmoid(negatives @ focus.T), focus)
                    temp_U[j] -= lr * gradient_u_pos
                    self.__U[negative_indices] -= lr * gradient_u_neg

                self.__U[t[i]] = temp_U
                self.__W[x[i]] -= lr * gradient_w

    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        all_words = []
        model = NearestNeighbors(n_neighbors=self.k, metric=metric).fit(self.__W)
        #
        for word in words:
            id = self.__w2i[word]
            context_vector = self.__W[id]
            if context_vector is not None:
                distance, indices = model.kneighbors([context_vector])
                # print(np.shape(distance))
                # print(np.shape(indices))
                closest_words = []
                for i in range(len(indices[0])):
                    index = indices[0][i]
                    dist = distance[0][i]
                    w = self.__i2w[index]
                    closest_words.append((w, dist))
                all_words.append(closest_words)

        return all_words

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v_shakespeare_50_2.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    if self.__i2w[i] == " ":
                        f.write(self.__i2w[i] + "+" + "+".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) + "+" + "+".join(map(lambda x: "{0:.6f}".format(x), self.__U[i, :])) + "\n")
                    elif self.__i2w[i] == "\\n" or self.__i2w[i] == "\n":
                        f.write("'\\" + repr(self.__i2w[i]) + "'" + "+" + "+".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) +  "+" + "+".join(map(lambda x: "{0:.6f}".format(x), self.__U[i, :])) + "\n")
                    else:
                        f.write(self.__i2w[i] + "+" + "+".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) + "+" +  "+".join(map(lambda x: "{0:.6f}".format(x), self.__U[i, :])) + "\n")
        except Exception as e:
            print(e)
            print("Error: failing to write model to the file")

    def write_U_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("u2v_shakespeare.txt", 'w') as f:
                W = self.__U
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    if self.__i2w[i] == " ":
                        f.write(self.__i2w[i] + "+" + "+".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) + "\n")
                    elif self.__i2w[i] == "\\n" or self.__i2w[i] == "\n":
                        f.write("'\\" + repr(self.__i2w[i]) + "'" + "+" + "+".join(
                            map(lambda x: "{0:.6f}".format(x), W[i, :])) + "\n")
                    else:
                        f.write(self.__i2w[i] + "+" + "+".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) + "\n")
        except Exception as e:
            print(e)
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                counter = 0
                for i, line in enumerate(f):
                    parts = line.split("+")
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)
                    counter += 1
                print(counter)

                w2v.init_params(W, w2i, i2w)
        except Exception as e:
            print(e)
            print("Error: failing to load the model to the file")
        return w2v

    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')
            # neighbors = self.find_nearest(text, 'euclidean')
            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.write_U_to_file()
        self.interact()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='shakespeare.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v_shakespeare_50_2.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=100, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
