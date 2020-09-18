### TEXT MODELING
# applying Alice Zhao's approach to text modeling and generation to motivational speeches

import requests
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.style.palettes import PALETTES, SEQUENCES, color_palette
from yellowbrick.style.rcmod import set_palette
from textblob import TextBlob
from gensim import matutils, models
import scipy.sparse
import logging
from nltk import word_tokenize, pos_tag
from collections import defaultdict
import random

# defining a function to scrape transcripts for motivational speakers
def url_to_transcript(url):
    '''Returns transcript data from jamesclear.com website'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    # iterating through each paragraph ('p') to find the desired text 
    text = [p.text for p in soup.find(class_="entry-content").find_all('p')]
    print(url)
    return text

# creating a list of urls for motivational speeches
urls = ['https://jamesclear.com/great-speeches/what-matters-more-than-your-talents-by-jeff-bezos',
       'https://jamesclear.com/great-speeches/the-danger-of-a-single-story-by-chimamanda-ngozi-adichie',
       'https://jamesclear.com/great-speeches/the-anatomy-of-trust-by-brene-brown',
       'https://jamesclear.com/great-speeches/2005-stanford-commencement-address-by-steve-jobs',
       'https://jamesclear.com/great-speeches/the-fringe-benefits-of-failure-and-the-importance-of-imagination-by-j-k-rowling',
       'https://jamesclear.com/great-speeches/do-schools-kill-creativity-by-ken-robinson',
       'https://jamesclear.com/great-speeches/the-multidisciplinary-approach-to-thinking-by-peter-kaufman',
       'https://jamesclear.com/great-speeches/your-elusive-creative-genius-by-elizabeth-gilbert',
       'https://jamesclear.com/great-speeches/make-good-art-by-neil-gaiman',
       'https://jamesclear.com/great-speeches/time-management-by-randy-pausch']

# creating a list of motivational speakers to match the urls
speakers = ['Jeff Bezos', 'Chimamanda Ngozi Adichie', 'Brené Brown', 'Steve Jobs', 
            'J.K.Rowling', 'Ken Robinson', 'Peter Kaufman', 'Elizabeth Glbert',
            'Neil Gaiman', 'Randy Pausch']

# scraping transcripts from each url
transcripts = [url_to_transcript(u) for u in urls]

# Make a new directory to hold the pickled text files
!mkdir transcripts

for i, s in enumerate(speakers):
    with open("transcripts/" + s + ".txt", "wb") as file:
        pickle.dump(transcripts[i], file)

# crreating a dictionary with speakers and transcripts
data = {}
for i, s in enumerate(speakers):
    with open("transcripts/" + s + ".txt", "rb") as file:
        data[s] = pickle.load(file)
        
# checking the dictionary and the content
data.keys()
data['Jeff Bezos'][:2]


# changing from 'list of texts' format to 'string' format for values
def combine_text(list_of_text):
    '''creates a string format from a list of texts'''
    combined_text = ''.join(list_of_text)
    return combined_text

# applying the function 'combine_text' to my data
data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

# creating a data frame from my dictionary
pd.set_option('max_colwidth',150) # setting the maximum width of columns

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df

# looking up the transcript of Chimamanda
data_df.transcript.loc['Chimamanda Ngozi Adichie']


## Text cleaning
# defining a custom function for cleaning the text
def clean_text(text):
    '''makes text lowercase, removes punctuation and removes words containing numbers'''
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n', '', text)
    return text

round1 = lambda x: clean_text(x)

data_clean = pd.DataFrame(data_df.transcript.apply(round1))
data_clean

data_clean.transcript.loc['Chimamanda Ngozi Adichie']


# Creating a document-term matrix using CountVectorizer and excluding English stop words
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm

# pickling the files
data_dtm.to_pickle("dtm.pkl")
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))

# open pickled file: document-term matrix
data = pd.read_pickle('dtm.pkl')
# flipping rows to columns and columns to rows
data = data.transpose()
data.head()

# finding top 30 words for each speaker
top_dict = {}
for s in data.columns:
    top = data[s].sort_values(ascending = False).head(30)
    top_dict[s] = list(zip(top.index, top.values))
    
top_dict

# printing the top 15 words for each speaker
for speaker, top_words in top_dict.items():
    print(speaker)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('-------------')

'''because some motivational speakers use the same words
we want to find unique words for each speaker'''

# saving common words in a list
words = []
for speaker in data.columns:
    top = [word for (word, count) in top_dict[speaker]]
    for t in top:
        words.append(t)

words

# which are the most common words
Counter(words).most_common()

# create a list with stop words if more than 5 speakers have used it
add_stop_words = [word for word, count in Counter(words).most_common() if count > 5]
add_stop_words

# add new common words to stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# create a new document-term matrix without the common words
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index
data_stop

pickle.dump(cv, open("cv_stop.pkl", "wb"))


# initializing
wc = WordCloud(stopwords=stop_words, background_color='white', colormap='Dark2',
               max_font_size=150, random_state=42)


# WordCloud for each motivational speaker
plt.rcParams['figure.figsize'] = [16, 6]

full_names = ['Jeff Bezos', 'Chimamanda Ngozi Adichie', 'Brené Brown', 'Steve Jobs', 
            'J.K.Rowling', 'Ken Robinson', 'Peter Kaufman', 'Elizabeth Glbert',
            'Neil Gaiman', 'Randy Pausch']

# Create subplots for each comedian
for index, speaker in enumerate(data.columns):
    wc.generate(data_clean.transcript[speaker])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index])
    
plt.show()

'''
Results:
Focus on time and persitance: Neil Gaiman, Elizabeth Gilbert, Brene Brown, Peter Kaufman
Focus on educaation and colledge: Ken Robinson and Randy Pausch
Focus on experience and failure: Steve Jobs
Focus on heritage and stories: Chimamanda Ngozi Adichie
Focus on gift: J.K.Rowling
Focus on trust: Jeff Bezos
'''


# Finding the number of unique words - words occuring only once
# not including words that occur zero times
unique_list = []
for speaker in data.columns:
    uniques = data[speaker].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# data frame with unique words
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['speaker', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
data_unique_sort

# plotting the number of unique words
y_pos = np.arange(len(data_words))
set_palette(palette='sns_pastel', color_codes=True)

plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.speaker)
plt.title('Number of Unique Words', fontsize=20)

plt.tight_layout()
plt.show()


## Sentiment Analysis - TextBlob

# creating lambda functions for polarity and subjectivitiy
# and adding the results in new columns
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data_df['polarity'] = data_df['transcript'].apply(pol)
data_df['subjectivity'] = data_df['transcript'].apply(sub)

# plotting the results

# adding new column with full names
speakers_alpha = ['Brené Brown', 'Chimamanda Ngozi Adichie', 'Elizabeth Glbert', 
            'J.K.Rowling', 'Jeff Bezos', 'Ken Robinson', 'Neil Gaiman', 'Peter Kaufman',
            'Randy Pausch', 'Steve Jobs']

data_df['speaker'] = speakers_alpha

# plot
plt.rcParams['figure.figsize'] = [10, 8]

for index, speaker in enumerate(data_df.index):
    x = data_df.polarity.loc[speaker]
    y = data_df.subjectivity.loc[speaker]
    plt.scatter(x, y, color='darkturquoise')
    plt.text(x+.001, y+.001, data_df['speaker'][index], fontsize=10)
    plt.xlim(-.01, .30) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()

'''
Results:
Chimamanda Ngozi Adiche uses the most objective and the least positive words
'''


## Topic Modeling

# for debugging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# preparing for LDA
topic_df = data_stop.transpose()
topic_df.head()

# gensim format: from sparse matrix to corpus
sparse_counts = scipy.sparse.csr_matrix(topic_df)
corpus = matutils.Sparse2Corpus(sparse_counts)

# terms and their location
cv = pickle.load(open("cv.pkl", "rb")) 
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)
lda.print_topics()

# Results are not conclusive, therefore:
# analyze only nouns
def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)

data_nouns = pd.DataFrame(data_clean.transcript.apply(nouns))

data_cvn = cv.fit_transform(data_nouns.transcript)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cv.get_feature_names())
data_dtmn.index = data_nouns.index

# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cv.vocabulary_.items())

# do the model again
ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=10)
ldan.print_topics()

# Results are again inconclusive, therefore:
# analyze nouns and adjectives
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)

data_nouns_adj = pd.DataFrame(data_clean.transcript.apply(nouns_adj))

cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.transcript)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
data_dtmna

# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())

ldana = models.LdaModel(corpus=corpusna, num_topics=3, id2word=id2wordna, passes=10)
ldana.print_topics()

'''
The results do not make much sense, probably because the motivational speeches
do not differ that much among each other.  
'''


## Text Generation
# extracting Jeff Bezos's text as an example
JB_text = data_df.transcript.loc['Jeff Bezos']

def markov_chain(text):
    '''The input is a string of text and the output will be a dictionary with each word as
       a key and each value as the list of words that come after the key in the text.'''
    
    # Tokenize the by splitting words and punctuation
    words = text.split(' ')
    
    # Initialize a default dictionary to hold all of the words and next words
    m_dict = defaultdict(list)
    
    # Create a zipped list of all of the word pairs and put them in word: list of next words format
    for current_word, next_word in zip(words[0:-1], words[1:]):
        m_dict[current_word].append(next_word)

    # Convert the default dict back into a dictionary
    m_dict = dict(m_dict)
    return m_dict

JB_dict = markov_chain(JB_text)

# generate text
def generate_sentence(chain, count=15):
    '''The input is a dictionary: key = current word, value = list of next words
       + the number of words I would like to see in my generated sentence.'''

    # Capitalize the first word
    word1 = random.choice(list(chain.keys()))
    sentence = word1.capitalize()

    # Generate the second word from the value list. Set the new word as the first word. Repeat.
    for i in range(count-1):
        word2 = random.choice(chain[word1])
        word1 = word2
        sentence += ' ' + word2

    # End it with a period
    sentence += '.'
    return(sentence)

generate_sentence(JB_dict)

# An example of the generated sentence:
# Month comes the back of cement-filled tires, a bunch of our choices. Build yourself a.


### End of analysis