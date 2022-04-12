import pandas as pd

data = pd.read_csv("test2 - Copy.csv")
#print(data.head())
#print("No. of rows: ", data.shape[0])
#print("No. of coloums: ", data.shape[1])
#print(data.info())
#print(data.Ad_Description.value_counts())

#####################################1. Data cleaning & pre-processing##################################
test_text = data["Ad_Description"][1]
print("***********************original text = \n",test_text)

##from bs4 import BeautifulSoup
##cleantext = BeautifulSoup(test_text, "lxml").text
##print(cleantext)

#_________________________________1.1 remove punct____________________________#
import re
test_text_no_punct = re.sub(r'[^\w\s]', ' ', test_text)
print("***********************punctuation removed = \n", test_text_no_punct)

#_________________________________1.2 remove stopwords____________________________#
import nltk
from nltk.corpus import stopwords

stop_words=stopwords.words('english')
additional_stopwds = ['youre', 'youve','youll','youd','shes','its','thatll','dont','shouldve','arent','couldnt',
                      'didnt','doesnt','hadnt','hasnt','havent','isnt','mightnt','mustnt','neednt','shant','shouldnt',
                      'wasnt','werent','wont','wouldnt','us','dont','would','sqm','sq','sqmtr','sqr','mtr','weve','theres',
                      'cant','th','sm','psm','whats','mtrs','thats','level','levels','meter','metre','sqmt','gst','pa','andor',
                      'however','today','call','available','ready','per','email','one','two','three','also','contact','approx',
                      'approximately','nla','gla','day','floor','floors','via','show','please','well','address','could',
                      'listing','id','still','touch','find','finding','found','able','st','yet','always','almost','although',
                      'among','anytime','call','rd','every','many','along','already','annum', 'become','come',
                      'either','etc','ever','everything','forward','whilst','whether','hesitate', 'regarding',
                      'looking', 'offering','instagram','offered','facebook','become','linkedin','sqmof',
                      'inspect','information','info','give','include','includes','including','advice','need','provide','provides',
                      'provided','providing','may','needs','must','love','unbiased','size','help']
stop_words.extend(additional_stopwds)

from nltk.tokenize import sent_tokenize, word_tokenize

word_tokens = word_tokenize(test_text_no_punct)
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#filtered_sentence = []
##for w in word_tokens:
##    if w not in stop_words:
##        filtered_sentence.append(w)
print("***********************word_tokens = \n:", word_tokens)
print("***********************filtered_sentence = \n:", filtered_sentence)



#_________________________________1.3 Lemmatization (convert to dictionary word)_______________________#
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in filtered_sentence:
    lemmatized_words.append(lemmatizer.lemmatize(word))
print("***********************lemmatized_words = \n:", lemmatized_words)    
lemma_filtered_sentences = ' '.join([str(item) for item in lemmatized_words])
print("***********************lemma_filtered_sentences = \n:", lemma_filtered_sentences)    
#print(filtered_sentences)
#tokenize_filtered_sentences = sent_tokenize(filtered_sentences)
#print(tokenize_filtered_sentences)

#_________________________________define a function of data_cleaner _______________________#
from tqdm import tqdm
def data_cleaner (data):
    clean_data = []
       for sentence in tqdm(data):
        cleantext = re.sub(r'[^\w\s]', ' ', cleantext)
        cleantext = [w for w in word_tokenize(cleantext) if not w.lower() in stop_words]
        cleantext = ' '.join([lemmatizer.lemmatize(w) for w in cleantext])
        clean_data.append(cleantext.strip())
    return clean_data
clean_data = data_cleaner(data.Ad_Description.values)

        
