import pandas as pd

df = pd.read_excel(r'news_headlines.xlsx')
headlines = pd.DataFrame(df, columns=['SENTENCES'])
print(headlines.head())
print()

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

stop_words = ['i' , 'me', 'myself', 'my', 'and', 'if', 'you' , ',']
def process_sentence(stopwords, headline):
    i =0
    while i < (len(headline)-1):
        for word in stopwords:
            if headline[i] == word:
                del headline[i]
                [headline.append(x) for x in headline if x not in headline] 
    return headline

def get_sentence_vector(tokenized_sentence):
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




corpus = []
for ind in df.index:
    if ind < 5:
        print(df['SENTENCES'][ind])

        sentence = df['SENTENCES'][ind]
        headline = [word.lower() for word in sentence.split()]
        print(headline)

        print()

        tokenized_headline = process_sentence(stopwords, headline)
        print('after removing stopwords:\n')
        print(tokenized_headlines)

        vectorized_sentence = get_sentece_vector(tokenized_sentence)
        

