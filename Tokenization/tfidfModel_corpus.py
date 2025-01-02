from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# # Ensure necessary NLTK data is 
# import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Read the text file
with open('Tokenization/holy_grail.txt', 'r') as file:
    script_text = file.read()

# Tokenize the text
tokens = word_tokenize(script_text)

# Remove non-alphabetic tokens
alpha_tokens = [token for token in tokens if token.isalpha()]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in alpha_tokens if token.lower() not in stop_words]

# Lemmatize the tokens
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

# Create a dictionary and corpus
dictionary = Dictionary([lemmatized_tokens])
corpus = [dictionary.doc2bow(lemmatized_tokens)]

# Create the TF-IDF model
tfidf_model = TfidfModel(corpus, smartirs='nnn')

# Calculate the TF-IDF weights
tfidf_weights = tfidf_model[corpus[0]]

print("TF-IDF Weights:", tfidf_weights)  # Debug print

# Print the first five TF-IDF weights
#notice that output is empty list because I only use one documents. 
# Tf-idf purpose is to compare multiple doctments.
print("First five TF-IDF weights:", tfidf_weights[:5])


"""Use Case Examples
- Search Engines: If a user searches for "Halt goes Arthur", Document 2 is likely to rank higher than Documents 1 and 3 because these terms have higher TF-IDF weights in Document 2.
- Topic Modeling: If you are clustering documents, the terms with high weights can act as features to distinguish the clusters.
- Summarization: Extract sentences or phrases that contain terms with the highest TF-IDF weights."""