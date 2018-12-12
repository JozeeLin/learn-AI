#https://github.com/nathanrooy/word2vec-from-scratch-with-python/blob/master/word2vec.py
import numpy as np
import re
from collections import defaultdict

class word2vec(object):
    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    #GENERATE TRAINING DATA
    def generate_training_data(self, settings, corpus):
        #GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word]+=1
        self.v_count = len(word_counts.keys())

        #generate lookup dictionaries
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        #CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            #CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):
                #w_target = sentence[i]
                w_target = self.word2onehot(sentence[i])

                #cycle through context window
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j<=sent_len-1 and j>=0:
                        w_context.append(self.word2onehot(sentence[j]))

                training_data.append([w_target, w_context])
        return np.array(training_data)

    def softmax(self, x):
        """softmax activation function"""
        e_x = np.exp(x-np.max(x))
        return e_x/e_x.sum(axis=0)

    def word2onehot(self, word):
        """convert word to one hot encoding"""
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    def forward_pass(self, x):
        """forward pass"""
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        print('h',h.shape)
        print('u',u.shape)
        print('y_c',y_c.shape)
        return y_c, h, u

    def backprop(self, e, h, x):
        #backpropagation
        dl_dw2 = np.outer(h,e)
        dh = np.dot(self.w2, e.T)
        dl_dw1 = np.outer(x,dh)

        #update weights
        self.w1 = self.w1-(self.eta*dl_dw1)
        self.w2 = self.w2-(self.eta*dl_dw2)

    def train(self, training_data):
        """TRAIN W2V model"""
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n)) #context matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count)) #embedding matrix

        #CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):
            self.loss = 0
            #CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:
                #forward pass
                y_pred, h, u = self.forward_pass(w_t)

                #calculate error
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                #backpropagation
                self.backprop(EI, h, w_t)

                #calculate loss
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c)*np.log(np.sum(np.exp(u)))
                self.loss += -2*np.log(len(w_c)) \
                            - np.sum([u[word.index(1)] for word in w_c]) \
                            +(len(w_c)*np.log(np.sum(np.exp(u))))

            print("EPOCH:",i,'LOSS:',self.loss)


    def word_vec(self, word):
        """input a word , return a vector (if available)"""
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def vec_sim(self, vec, top_n):
        """input a vector, returns nearest word(s)"""

        #cycle through vocab
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec)*np.linalg.norm(v_w2)
            theta = theta_num/theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda(word, sim):sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print word, sim


    def word_sim(self, word, top_n):
        """input word, returns top [n] most similar words"""
        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        #cycle through vocab
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1)*np.linalg.norm(v_w2)
            theta = theta_num/theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda(word, sim):sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print word, sim


settings = {}
settings['n'] = 5 #dimension of word embeddings
settings['window_size'] = 2 #context window +/- center word
settings['min_count'] = 0 #minimum word count
settings['epochs'] = 5000 #number of training epochs
settings['neg_samp'] = 10 #number of negative words to use during training
settings['learning_rate'] = 0.01 #learning rate
np.random.seed(0)  #set the seed for reproducibility

corpus = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]

# INITIALIZE W2V MODEL
w2v = word2vec()

# generate training data
training_data = w2v.generate_training_data(settings, corpus)

# train word2vec model
w2v.train(training_data)
