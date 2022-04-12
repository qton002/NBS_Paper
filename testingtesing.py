import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

text='Building undergoing ground floor renovation incorportate new retail offering new lobby café much NBS Frank take proactive approach tenant understanding need provide range suitable commercial property option Contact experience refreshing approach commercial real estate Great natural lighting existing fitout Includes reception x office open plan area enclosed kitchen Agrade building fantastic view Close proximitiy everything need Onsite car parking available Building undergoing ground floor renovation incorportate new retail offering new lobby café much NBS Frank take proactive approach tenant understanding need provide range suitable commercial property option Contact experience refreshing approach commercial real estate Great natural lighting existing fitout Includes reception x meetingboardrooms x open plan area enclosed kitchen Agrade building Close proximitiy everything need Onsite car parking available Building undergoing ground floor renovation incorportate '
words = nltk.word_tokenize(text)

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in words:
    lemmatized_words.append(lemmatizer.lemmatize(word)) ##lemmatizer words into meaningful dictionary form

print(lemmatized_words)
