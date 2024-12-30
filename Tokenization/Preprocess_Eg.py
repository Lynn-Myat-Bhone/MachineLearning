from nltk.stem import WordNetLemmatizer
from collections import Counter

stop_words = ['the','and','a','an']

#Read the text file
with open('Tokenization/holy_grail.txt','r')as file :
    holy_grail = file.read()
    
words = holy_grail.split()

#retail words contain alphabets only
alpha_only = [t for t in words if t.isalpha()]

#remove the stop words
no_stop_words = [n for n in alpha_only if n not in stop_words ]

wordsLemma = WordNetLemmatizer()
# Lemmatize all tokens into a new list
lemma = [wordsLemma.lemmatize(i) for i in no_stop_words ]

#create the bag of words
bow = Counter(lemma)
print(bow.most_common(10))
