import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

##-----------------------------------------------open files-----------------------------------------------
#filename = "sent_tokenized_true_only.txt"  
filename = "test2.csv"  
f = open(filename,"r",encoding="utf-8")
example_text = f.read()
example_text = example_text.replace("\t", " ")
##print(example_text)
##-----------------------------------------------remove punctuations and numbers-----------------------------------------------
punct_N_num = '''*-.,~\/&#+=;<>'’‘{}[]()?:@_$^|·•"—0123456789%–“”…²!'''
no_punct_N_num = ""
for char in example_text:
    if(char not in punct_N_num):
        no_punct_N_num = no_punct_N_num + char
##print(no_punct_N_num)

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
                      'among','anytime','call','rd']
                      #'every','many','along','already','annum', 'become','come','either','etc','ever','everything','forward','whilst','whether'] want to add, but don't want to re-run.
stop_words.extend(additional_stopwds)
words_no_stopwds= [w for w in words if not w.lower() in stop_words]
lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in words_no_stopwds:
    lemmatized_words.append(lemmatizer.lemmatize(word))
#print(lemmatized_words)    
filtered_sentences = ' '.join([str(item) for item in lemmatized_words])
#print(filtered_sentences)
#tokenize_filtered_sentences = sent_tokenize(filtered_sentences)
#print(tokenize_filtered_sentences)

##-----------------------------------------------Bag of Words https://www.youtube.com/watch?v=8JcLENGoXL0&list=PLP_4EPVEox99f1-_JMRpQbPkcGlMd5Xoq&index=3----------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
#from feature_importance import FeatureImportance
#feature_importance = FeatureImportance()

count_vec = CountVectorizer()
count_occurs = count_vec.fit_transform([filtered_sentences])
#print(count_occurs)
wd_freq = count_occurs.toarray()
#print(wd_freq)
feature_names = count_vec.get_feature_names_out() ##in array
#print(feature_names)
feature_names_txt ='\n'.join([str(item) for item in feature_names])  ##in str
#print(feature_names_txt)

##ps = PorterStemmer()
##for w in feature_names:
##    print(ps.stem(w))

count_occur_df = pd.DataFrame(
    (count, word) for word, count in
    zip(wd_freq.tolist()[0],
        feature_names))
count_occur_df.columns = ['Word','Word_Freq']
count_occur_df.sort_values('Word_Freq', ascending=False,inplace=True)
pd.set_option("display.max_rows", None, "display.max_columns", None) ##set this to display all results
#count_occur_df.head()
#print(count_occur_df)
#print(count_occur_df.head())
#print(wd_freq)

##data = load_iris()
##display(df.to_string())

count_vec2 = CountVectorizer(analyzer='word', ngram_range=(1,3))
count_occurs2 = count_vec2.fit_transform([filtered_sentences])
feature_names2 = count_vec2.get_feature_names() ##in array
feature_names_txt2 ='\n'.join([str(item) for item in feature_names2])  ##in str
#print(feature_names_txt2)

count_occur_df2 = pd.DataFrame(
    (count, word) for word, count in
    zip(count_occurs2.toarray().tolist()[0],
    count_vec2.get_feature_names_out()))
count_occur_df2.columns = ['Word2','Word_Freq2']
count_occur_df2.sort_values('Word_Freq2', ascending=False,inplace=True)
count_occur_df2.head()
print(count_occur_df2)
#print(count_occur_df2.head())


##https://www.youtube.com/watch?v=8JcLENGoXL0&list=PLP_4EPVEox99f1-_JMRpQbPkcGlMd5Xoq&index=3

f.close()












