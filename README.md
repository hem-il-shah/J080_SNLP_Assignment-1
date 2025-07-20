# J080_SNLP_Assignment-1
## Name: Hemil Shah
## Roll Number: J080
## SAP Id:70092300088

# Word2Vec and Movie Review Sentiment Analysis Assignment

This repository documents two main NLP experiments:  
1. Exploring pretrained word2vec models and analogy tasks  
2. Building a movie review sentiment classifier using various word vector approaches

## Part 1: Exploring Pretrained Word2Vec Embeddings

### Setup

We use the **word2vec-google-news-300** pretrained model (3 million words/phrases, 300-dim vectors, trained on 100 billion words from Google News)[1][9].

#### Loading the Model

```python
import gensim.downloader as api
wv_pretrained = api.load("word2vec-google-news-300")
```

### 1. Most Similar Words

Pick 5 sample words and retrieve the most similar terms for each.

```python
words = ["dog", "computer", "music", "happy", "city"]
for word in words:
    print(f"Most similar to '{word}':")
    print(wv_pretrained.most_similar(word)[:5])  # Show top 5
```
Sample output:
- **dog**: puppy, dogs, pooch, pup, canine
- **computer**: computers, PC, laptop, software, hardware
- **music**: musical, musician, songs, song, musicians
- **happy**: glad, joyous, pleased, joyful, delighted
- **city**: town, cities, suburb, downtown, metropolis

### 2. Word Vector Arithmetic (Analogies)

Replicate the famous analogy and add 2-3 new ones:

```python
# Classic example
wv_pretrained.most_similar(positive=["king", "woman"], negative=["man"])

# Additional analogies
wv_pretrained.most_similar(positive=["Paris", "Italy"], negative=["France"])
wv_pretrained.most_similar(positive=["cat", "puppy"], negative=["kitten"])
wv_pretrained.most_similar(positive=["walking", "swam"], negative=["walk"])
```

Example results:
- **king - man + woman ≈ queen**
- **Paris - France + Italy ≈ Rome**
- **cat - kitten + puppy ≈ dog**
- **walking - walk + swam ≈ swum**

*Note: actual output may vary slightly due to internal Gensim implementation and floating-point precision.*[5][7]

## Part 2: Movie Review Sentiment Classifier Using Word Vectors

### Dataset

Use the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data).

### 1. Text EDA & Cleaning

**Tasks:**
- Check class distribution (positive/negative)
- Plot review length distributions, frequent words, etc.
- Clean text:
  - Lowercase transformation
  - Remove punctuation
  - Remove stopwords
  - (Optional) Lemmatization/stemming

```python
import pandas as pd
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return ' '.join([w for w in text.split() if w not in stop_words])
df['clean_review'] = df['review'].apply(clean_text)
```

### 2. Feature Engineering

#### Vector Representations
- **Pretrained word2vec vectors**: Average of word vectors for each review
- **Custom Skip-gram Word2Vec**
- **Custom CBOW Word2Vec**
- **Custom FastText vectors**

Sample training for custom vector models:

```python
from gensim.models import Word2Vec, FastText

# Tokenize reviews
sentences = [rev.split() for rev in df['clean_review'].tolist()]

# Skip-gram
sg_model = Word2Vec(sentences, sg=1, vector_size=100, window=3, min_count=1, workers=4).wv

# CBOW
cbow_model = Word2Vec(sentences, sg=0, vector_size=100, window=3, min_count=1, workers=4).wv

# FastText
fasttext_model = FastText(sentences, sg=1, vector_size=100, window=3, min_count=1, workers=4).wv
```
*Refer to Gensim's documentation for more options and details*[3][4].

### 3. Training Sentiment Classifier

Use ML models like Logistic Regression, SVM, or simple neural nets. For each review:
- Vectorize (average word vectors per review)
- Train and evaluate the classifier for each embedding setup

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example: X_train, X_test: averaged document vectors; y_train, y_test: labels

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

### 4. Performance Comparison

Tabulate accuracy (or other metrics) for each vectorization method:

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| **Pretrained W2V**    |   0.XX   |   0.XX    |  0.XX  |  0.XX    |
| **Custom Skip-gram**  |   0.XX   |   0.XX    |  0.XX  |  0.XX    |
| **Custom CBOW**       |   0.XX   |   0.XX    |  0.XX  |  0.XX    |
| **Custom FastText**   |   0.XX   |   0.XX    |  0.XX  |  0.XX    |

(*Fill in scores after your experiments*)

## References

- [1] word2vec-google-news-300 Explained
- [3] Gensim Word2Vec Documentation
- [4] TensorFlow Word2Vec Tutorial
- [5] Gensim .most_similar() Method
- [7] Community Example on .most_similar()
