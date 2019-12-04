# News-headlines-clustering
this project is about clustering news headlines from a corpus into various categories like sports, polical, entertainment etc.
I have used python programming language and various libraries like pandas, numpy and matplotlib. For obtaining the word embeddings of these headlines, I have used gensim's FastText pre-trained word embedding model for training word vectors. Size of these embeddings is (100,)
Fastext used CBOW for default with a window size of 5 and alsp provides word embeddings for such words which are not in the vocabulary provided at least 1 char grams can be found in the vocab. 
The Fastest model used in this project is trained over Lee Corps.
