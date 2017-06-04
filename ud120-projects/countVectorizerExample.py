from sklearn.feature_extraction.text import CountVectorizer

#create the vectorizer

vectorizer = CountVectorizer()



string1 = "hi Katie the self driving car will be late Best Sebastian"
string2 = "Hi Sebastion the machine learning class will bre great great great Best Katie"

string3 = "Hi Katie the machine learning class will be most excellent"
email_list = [string1,string2,string3]

#need to fit the data
bag_of_words=vectorizer.fit(email_list)

#must also transform the data

bagOW = vectorizer.transform(email_list)



print(bagOW)
document1 = string2.split()
print("The 7th word of the first doc is", document1[6])


print(vectorizer.vocabulary_.get("great"))