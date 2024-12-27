import re
import matplotlib.pyplot as plt
from nltk.tokenize import regexp_tokenize

# Read the script from a text file
with open('Tokenization\holy_grail.txt', 'r') as file:
    holy_grail = file.read()
# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]
print(lines)
# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s,'\w+') for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)

# Show the plot
plt.show()