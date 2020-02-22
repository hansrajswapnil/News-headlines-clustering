import numpy as np
from collections import defaultdict


class word2vec():
    def __init__(self):
        self.n = settings['n']  #dim of word embeddings, also refer to size of hidden layers
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']    #num of training epochs
        self.window = settings['window_size']   #context window +- center word

    
    def generate_training_data(self, settings, corpus):
        word_count = defaultdict(int)  #finds unq word counts using dictionary
        
        for row in corpus:
            for word in row:
                word_count[word] += 1

        self.v_count = len(word_count.keys())   #how many unq words in the dict
        
        self.word_list = list(word_count.keys())    #generatings look-up dict(vocab)
        
        #generate word:index
        self.word_index = dict((word, i) for i, word in enumerate(self.word_list))

        #generate index:word
        self.index_word = dict((i, word) for i, word in enumerate(self.word_list))


        training_data = []

        #cycling through each sentence in corpus
        for sentence in corpus:
            sent_len = len(sentence)
            
            #cyle through each word in sentence
            for i, word in enumerate(sentence):

                #converting target word into one-hot enc
                w_target = self.word2onehot(sentence[i])

                w_context= []

                #cycle through context window
                for j in range(i-self.window, i+ self.window+1):
                    #criteria for context word
                    #1. target word != context word
                    #2. index must be >= 0 (j >= 0)
                    #3. index must be <= len(sentence)

                    if j != i and j<=sent_len-1 and j>=0:
                        
                        #append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))

                
                training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):

        #initialize a blank vector __ word_vec
        word_vec = [0 for i in range(0, self.v_count)]

        #get the ID of the word from word_index
        word_index = self.word_index[word]
        
        #change value to 1 acc to ID of the word
        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):
        
        #initialize weight matrix
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))      #9x10
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))      #10x9


        #cycle through each epoch
        for i in range(self.epochs):
            self.loss = 0       #initialize loss to 0

            #cycle through each training example
            for w_t, w_c in training_data:      #w_t is the target vector & w_c is the context vector

                #forward pass
                y_pred, h, u = self.forward_pass(w_t)
                print('\nvector for target word: ',w_t)
                print('\n\nw1-before backprop\n',self.w1)
                print('\n\nw2-before backprop\n',self.w2)


                #calculate error
                #for a target word, cal diff b/w y_pred & each of the context words
                #sum up the diffreneces for each target word
                EI = np.sum([np.subtract(y_pred,word) for word in w_c],axis=0)

                print('\n\n')
                print(EI.shape)
                print('Error\n',EI)

                #backpropagation
                #we use SGD to backpropagate errors - cal loss on the output layer
                self.backprop(EI, h, w_t)

                print('\n\nW1-after backprop\n',self.w1)
                print('\n\nW2-after backprop\n',self.w2)


                #calculate loss
                self.loss += -np.sum([u[word_index(1)] for word in w_c ]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print('\nEpoch:', i, 'Loss: ', self.loss)

    def forward_pass(self, x) :
        
        #x is the one-hot encoding for w_t, shape - 9x1
        #passing through 1st hidden matrix-----10x9

        h = np.dot(self.w1.T, x )
        #h ----- 10x1
        #dot product with second layer hidden matrix-----10x9
        u = np.dot(self.w2.T, h)
        #u ----- 9x1

        #run output layer through softmax
        y_c = self.softmax(u)

        return y_c, h, u

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x/e_x.sum(axis=0)

    def backprop(self, e, h, x):
        d1_dw2 = np.outer( h, e)     #d1_dw2 : (9x1) X (10x1)
        d1_dw1 = np.outer(x, np.dot(self.w2, e))      #x --- 9x1 ; self.w2 --- 10x9, e.T --- 1x9, e --- 9x1, self.w2.T --- 10x9
        #d1_dw1 : 9x10
        #update weights
        self.w1 = self.w1 - (self.lr * d1_dw1)
        self.w2 = self.w2 - (self.lr * d1_dw2)

    #get vector from the word
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    #input vector retunr nearest word(s)
    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            #find the similarity score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta


        word_sorted = sorted(word_sim.items(), key=lambda kv : kv[1], reverse=True)

        for word, sim in word_sorted[:top_n]:
            print('\n')
            print(word, sim)


text = 'natural language processing and machine learning is fun and exciting'

corpus = [[word.lower() for word in text.split()]]

settings = {
        'window_size':2,
        'n':10,
        'epochs':50,
        'learning_rate':0.01
        }

#initialize object
w2v = word2vec()

#numpy ndarray with one-hot representation for (target_word, context_words)
training_data = w2v.generate_training_data(settings, corpus)

# Training
w2v.train(training_data)

# Get vector for word
word = "machine"
vec = w2v.word_vec(word)
print(word, vec)

# Find similar words
w2v.vec_sim("machine", 3)

