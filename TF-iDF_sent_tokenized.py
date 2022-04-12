import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

##-----------------------------------------------open files-----------------------------------------------
#filename = "sent_tokenized_true_only.txt"  
filename = "test2.csv"  
f = open(filename,"r",encoding="utf-8")
example_text = f.read()
#example_text = example_text.replace("\t", " ")
#print(example_text)

##-----------------------------------------------remove punctuations and numbers-----------------------------------------------
punct_N_num = '''*-.,~\/&#+=;<>'’‘{}[]()?:@_$^|·•"—0123456789%–“”…²!'''
no_punct_N_num = ""
for char in example_text:
    if(char not in punct_N_num):
        no_punct_N_num = no_punct_N_num + char
#print(no_punct_N_num)

##-----------------------------------------------sent_tokenize, word_tokenize-----------------------------------------------
sentences = sent_tokenize(no_punct_N_num)
words = nltk.word_tokenize(no_punct_N_num)
##sentences = sent_tokenize(example_text)
##words = nltk.word_tokenize(example_text)
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
                      'provided','providing','may','needs','must',
                      'love','unbiased','size','help']
stop_words.extend(additional_stopwds)
words_no_stopwds= [w for w in words if not w.lower() in stop_words]
#print(words_no_stopwds)
lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in words_no_stopwds:
    lemmatized_words.append(lemmatizer.lemmatize(word))
#print(lemmatized_words)    
filtered_sentences = ' '.join([str(item) for item in lemmatized_words])
#print(filtered_sentences)
#tokenize_filtered_sentences = sent_tokenize(filtered_sentences)
#print(tokenize_filtered_sentences)

##----------------------------------------------TF-iDF----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)

TfidfVectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2,3))
vectors = TfidfVectorizer.fit_transform([filtered_sentences])
#print(vectors)
wd_freq = vectors.toarray()
#print(wd_freq)
feature_names_vectors = TfidfVectorizer.get_feature_names() ##in array
#print(feature_names_vectors)

vectors_df = pd.DataFrame(
    (count, word) for word, count in
    zip(wd_freq.tolist()[0],
        feature_names_vectors))
vectors_df.columns = ['Words','TF_iDF']
vectors_df.sort_values('TF_iDF', ascending=False,inplace=True)
pd.set_option("display.max_rows", None, "display.max_columns", None) ##set this to display all results
#print(vectors_df.head())
print(vectors_df)

##https://www.youtube.com/watch?v=8JcLENGoXL0&list=PLP_4EPVEox99f1-_JMRpQbPkcGlMd5Xoq&index=3
f.close()












