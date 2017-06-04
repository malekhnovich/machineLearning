#!/usr/bin/python

import os
import pickle
import re
import sys
import unicodedata

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer


"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1

        path = os.path.join('..', path[:-1])
        #print path
        email = open(path, "r")


        ### use parseOutText to extract the text from the opened email
        stemmed_string = parseOutText(email)
        bannedList = ["sara","shackleton","chris","germani","sshacklensf"]
        for word in bannedList:
            if word in stemmed_string:
                new_stemmed_string = word.replace(word,"")

        #print(new_stemmed_string)
        word_data.append(new_stemmed_string)



        #print(from_person)
        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]

        ### append the text to word_data

        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris

        email.close()
        if name=='sara':
            from_data.append(0)
        elif name=='chris':
            from_data.append(1)
        #print(name)
print("word data index of 152 is ",word_data[152])
print "emails processed"
from_sara.close()
from_chris.close()






pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )





### in Part 4, do TfIdf vectorization here


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')



matrix =vectorizer.fit(word_data)

vocab = matrix.get_feature_names()

#print("vocab equals",vocab)
print("The unique elements in vocabluary is",len(vocab))
#vocab = unicodedata.normalize('NFKD', vocab).encode('ascii', 'ignore')
print("word data is a ",type(vocab))
print("Word at index vocab[34597]",vocab[34597])