from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
nltk.download('words')
with open('Tokenization/holy_grail.txt','r') as file:
    holy_griail = file.read() 
    


sentences = sent_tokenize(holy_griail)
tokens = [word_tokenize(i) for i in sentences]
#remove non-alphabetical words
alpha_token = [[word for word in sentence if word.isalpha()] for sentence in tokens]

#tag each token into parts of speech
pos_sentences = [nltk.pos_tag(i) for i in alpha_token]

# Create the named entity chunks
Chunks = nltk.ne_chunk_sents(pos_sentences, binary=True)

for i in Chunks:
    for j in i:
        if hasattr(j, "label") and j.label() == "NE":
            print(j)