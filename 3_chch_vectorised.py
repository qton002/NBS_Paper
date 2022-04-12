import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import nltk.corpus as Corpus
import numpy as np

filename = "3_chch_test2.csv"  ##spelling check in Excel
f = open(filename,"r",encoding="utf-8")
ad_descriptions = f.read()

ad_descriptions = re.sub(r'[\*\;\•]', '\n', ad_descriptions) 
sent_text = sent_tokenize(ad_descriptions)
sentences_arrays = [sentence.split('\n') for sentence in sent_text]
sentences = []
for sub_list in sentences_arrays:
    for sentence in sub_list:
        sentences.append(sentence.strip())
#print("sentences: ", sentences)

vocab=nltk.FreqDist(sentences)
#print("Length of vocab: ", len(vocab))
freq_tuple=vocab.most_common(200)
freq_array = np.asarray(freq_tuple)
#print("the most freq words (200) freq_array:", freq_array) #help to remove agency_related contents
#print("vocab.most_common(200):", vocab.most_common(200)) #help to remove agency_related contents

##import string
##print("string.punctuation:", string.punctuation)
##freq_array_noPunct = [''.join(letter for letter in word if letter not in string.punctuation) for word in freq_array]
##print("the most freq words (200) freq_array_noPunct:", freq_array_noPunct) #help to remove agency_related contents

df_most_common = pd.DataFrame(freq_array)
df_most_common.columns=["most_freq","freq"]
df_most_common.to_csv('3_chch_most_common200.csv', encoding='utf-8') 

df = pd.DataFrame(sentences)
df.columns = ["Ad_Desc"]
df .to_csv('3_chch_sent_text.csv', encoding='utf-8', index = False, header=False) #save to new file. Manully classify the words.

###____________________________________________________####

f.close()


###____________________________________________________####
filename_sent_text = "3_chch_sent_text.csv"  
with open(filename_sent_text,"r",encoding="utf-8") as file:
    mystring = file.readlines()
    for i,line in enumerate(mystring):
        for pattern in [ 'Features:',
                         'http://www.'
                         'collier.co.nz',
                         '.co.nz',
                         'Show more"',
                         'We recommend the tenant seeks their own expert assessment from a suitably qualified professional."'
                         'Contact the agent today to view this exciting new way to work!',
                         'Call to view today!"',
                         'Features include:',
                         'Property Features:',
                         'For more details or to arrange a viewing contact Colin Barratt on 027 528 3077."'
                         'Design Features',
                         'Christchurch Commercial Real Estate Limited',
                         'Licensed Agent REAA 2008',
                         'Tell us your requirements, the landlord is prepared to negotiate.',
                         'Key features:',
                         'For Lease By Negotiation"',
                         'Call Paula Raine 027 221 4997 for further information.',
                         'Available now',
                         'FEATURES>>>',
                         'Call, Text or Email Bill Riding on 021 743 464 or 03 961 4000 Today.'
                         'Rarely do opportunities like this come available so contact the listing agent today.'
                         'Register your interest today as it may be possible at this stage to incorporate any requirements you may have.'
                         'Price for Lease: By Negotiation + GST + Outgoings"',
                         'Available for occupation now.'
                     
        
                        ]:
            if pattern in line:
                mystring[i] = line.replace(pattern,"")                    
    # print the processed lines
    remove_agent_related_info = "".join(mystring)
    #print("remove_agent_related_info: ",remove_agent_related_info) #agency_related contents removed as much as possible

cleantext = re.sub(r'[^\w\s]', ' ', remove_agent_related_info) #remove if ^not \w(letters, numbers) + \s(whitespaces)
#print("cleantext:",cleantext)

words = word_tokenize(cleantext) #tokenize words
#print("words:",words)

cleantext_no_digits = ' '.join([w for w in word_tokenize(cleantext) if w.isalpha()]) #remove all digits
#print("cleantext_no_digits:",cleantext_no_digits)

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
                      'provided','providing','may','needs','must','love','unbiased','size','help','negotiation','metrocommercial','www','http','arrange','viewing']
stop_words.extend(additional_stopwds)
cleantext_no_stopwds = [w for w in word_tokenize(cleantext_no_digits) if not w.lower() in stop_words] #remove stopwords
#print("cleantext_no_stopwds:",cleantext_no_stopwds)

lemmatizer = WordNetLemmatizer()
cleantext_lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in cleantext_no_stopwds]) #covert a word to its base form. e.g. walks to walk
#print("cleantext_lemmatized:",cleantext_lemmatized)

## try to print the most freq words, but not working
##myCorpus = Corpus([lemmatizer.lemmatize(w) for w in cleantext_no_stopwds])
##Most_freq_100 = nltk.FreqDist(myCorpus)
##print("Most_freq_100:",Most_freq_100.most_common(100))
##myCorpus.dispersion_plot(Most_freq_100)

##-----------------------------------------------Bag of Words https://www.youtube.com/watch?v=8JcLENGoXL0&list=PLP_4EPVEox99f1-_JMRpQbPkcGlMd5Xoq&index=3----------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
pd.set_option("display.max_rows", None, "display.max_columns", None) ##set this to display all results
count_vec = CountVectorizer(analyzer='word', ngram_range=(2,3))
count_occurs = count_vec.fit_transform([cleantext_lemmatized])
feature_names = count_vec.get_feature_names() ##in array
feature_names_txt ='\n'.join([str(item) for item in feature_names])  ##in str
#print(feature_names_txt2)

count_occur_df = pd.DataFrame(
    (count, word) for word, count in
    zip(count_occurs.toarray().tolist()[0],
    count_vec.get_feature_names_out()))
count_occur_df.columns = ['Word2','Word_Freq']
count_occur_df.sort_values('Word_Freq', ascending=False,inplace=True)

#print("count_occur_df: ",count_occur_df)
#print(count_occur_df2.head())

count_occur_df.to_csv('3_chch_BagofWords.csv', encoding='utf-8', index=False)

##----------------------------------------------TF-iDF----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option("display.max_rows", None, "display.max_columns", None) ##set this to display all results

TfidfVectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2,3))
vectors = TfidfVectorizer.fit_transform([cleantext_lemmatized])
#print("vectors", vectors)
wd_freq = vectors.toarray()
#print("wd_freq: ", wd_freq)
feature_names_vectors = TfidfVectorizer.get_feature_names() ##in array
#print("feature_names_vectors: ",feature_names_vectors)

vectors_df = pd.DataFrame(
    (count, word) for word, count in
    zip(wd_freq.tolist()[0],
        feature_names_vectors))
vectors_df.columns = ['Words','TF_iDF']
vectors_df.sort_values('TF_iDF', ascending=False,inplace=True)

#print(vectors_df.head())
#print("vectors_df:", vectors_df)

vectors_df.to_csv('3_chch_TF_iDF_keywords.csv', encoding='utf-8', index=False)




##----------------------------------------------Text Classification [Naive Bayes Classifier]----------------------------------------------
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




file.close()

print("3_chch_vectorised run complete 100%+++++++++++++++++++++++++++++++++++++")


