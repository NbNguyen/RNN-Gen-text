import nltk

with open('data/sentence.txt', 'r') as f:
    for line in f:
        sentence = line.split('/')
        words = [nltk.word_tokenize("".join(sentence))]
        print sentence
        print words
        print ""