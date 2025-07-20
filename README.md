# J080_SNLP_Assignment-1
## Name: Hemil Shah
## Roll Number: J080
## SAP Id:70092300088

Assignment
Part 1
Use a pretrained word2vec model (example - word2vec-google-news-300 )
Pick any 5 words of your choice and find the model similar words for each of these 5 words.
Just like the experiment from the lecture where we checked king - man + woman ~= queen  - come up with 2-3 similar examples and test them with the pre-trained word2vec model’s vectors.
Part 2
Build a movie review sentiment classifier using WordVectors
Dataset: IMDB Dataset of 50K Movie Reviews​
Tasks:
Perform text EDA
Clean the text - remove noisy tokens like punctuations and stopwords
Train an ML model of your choice using:
A pre-trained W2V model’s vector (pick any model from the web)
Custom Skip-gram vectors
Custom CBoW vectors
Custom FastText vectors
Tabulate the model performance stats.
Additional resources
Loading a W2V model from the web:
import gensim.downloader as api

# load pretrained model
wv_pretrained = api.load("...")

# find most similar words
wv_pretrained.most_similar(
	positive=["king", "woman"], 
	negative=["man"]
)
​
Creating a custom W2V model
from gensim.models import Word2Vec

custom_wv_model = Word2Vec(
    sentences=[...], # list of list of tokens
    sg=1, # 1 = skipgram, 0 = CBoW
    vector_size=100, # embedding size
    window=3, # sliding window size
    min_count=1, # minimum frequency of a word to be considered for training
    workers=4, # cpu workers
).wv # access the main wv model
​
Creating a custom FastText model
from gensim.models import FastText

custom_fasttext_model = FastText(
    sentences=[...], # list of list of tokens
    sg=1, # 1 = skipgram, 0 = CBoW
    vector_size=100, # embedding size
    window=3, # sliding window size
    min_count=1, # minimum frequency of a word to be considered for training
    workers=4, # cpu workers
).wv # access the main wv model
