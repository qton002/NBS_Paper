import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import nltk.corpus as Corpus
import numpy as np

filename = "2_wlg_test2.csv"  ##spelling check in Excel
f = open(filename,"r",encoding="utf-8")
ad_descriptions = f.read()

#ad_descriptions = re.sub(r'[\*\;]', '\n', ad_descriptions) #don't need this for chch
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

df_most_common = pd.DataFrame(freq_array)
df_most_common.columns=["most_freq","freq"]
df_most_common.to_csv('2_wlg_most_common200.csv', encoding='utf-8')

df = pd.DataFrame(sentences)
df.columns = ["Ad_Desc"]
df .to_csv('2_wlg_sent_text.csv', encoding='utf-8', index = False, header=False) #save to new file. Manully classify the words.

###____________________________________________________####

f.close()


###____________________________________________________####
filename_sent_text = "2_wlg_sent_text.csv"  
with open(filename_sent_text,"r",encoding="utf-8") as file:
    mystring = file.readlines()
    for i,line in enumerate(mystring):
        for pattern in ['Cell: 027 4411 015 Office: 471 2364 or email: tom.burke@capitalrealty.co.nz',
                        'Deal with a professional office leasing broker.',
                        'Contact Tom Burke for full details of this listing & more private listings.',
                        'Any prices quoted are accurate at the time of listing and can be confirmed on receipt of your enquiry.',
                        'Features:',
                        'Cell: 027 4411 015 Office: 471 2364 or email: tom.burke@capitalrealty.co.nz  Note: we have access to all listings incl sole listings on the market.',
                        'Quick Easy Search & Market Report available at www.capitalrealty.co.nz"',
                        'Contact Tom Burke over 20 years’ experience & over 450 leasing deals for full details of this listing & more private listings in the area.',
                        'Contact Tom Burke (over 20 years’ experience & over 450 leasing deals) for full details of this listing & more private listings in the area.',
                        'Contact Tom Burke (over 20 years’ experience & over 450 leasing deals) for full details of this listing ',
                        '"Contact the leasing agent, Tom Burke (over 20 years’ experience & over 450 leasing deals) for full details of this listing and more private listings and to view this tenancy."',
                        'Quick Easy Search & Blog/News & Market Report available at www.capitalrealty.co.nz"',
                        '20 years experience and over 450 deals.',
                        '(18 year’s experience and over 450 deals.)',
                        'For market news and more, go to www.capitalrealty.co.nz  Our guarantee: We will never knowingly advertise a listing that is no longer available."',
                        '"For market news and more, go to www.capitalrealty.co.nz  ""'
                        'Contact Tom Burke over 20 years’ experience & over 450 leasing deals for full details of this listing & more private listings.',
                        '& more private listings.',
                        'Quick Easy Search & Blog/News & Market Report all available at www.capitalrealty.co.nz"',
                        'Show more"',
                        'Contact the leasing agent, Tom Burke over 20 years’ experience & over 450 leasing deals for full details of this listing and more private listings and to view this tenancy.',
                        'Contact Tom Burke (over 20 years’ experience & over 450 deals) for full details of this listing ',
                        'Cell: 027 4411 015 Office: 471 2364 or email tom.burke@capitalrealty.co.nz',
                        'For market news and more, go to www.capitalrealty.co.nz"',
                        'FEATURES:',
                        'If any problem is found, it is attended to promptly.',
                        "*** Not what you're hunting for?",
                        'For market news and more, go to www.capitalrealty.co.nz',
                        'Our guarantee: We will never knowingly advertise a listing that is no longer available."',
                        '18 years experience and over 450 deals.',
                        'Contact Tom Burke over 25 years’ experience & over 500 leasing deals for full details of this listing & more private listings in the area.',
                        '"BUILDING DETAILS:',
                        "Can't find the right space for you?",
                        'If you\'d like to see the full range of Wellington CBD or fringe spaces that meet your office relocation requirements submit a brief at www.wellingtonofficespace.co.nz"',
                        'For further details or to view, call Calder today!"',
                        'We currently have other options available in the Wellington CBD and city fringe.',
                        'Contact us for help finding the right environment for you and your business."',
                        'Mark Melville 022 154 6558 - mailto:mark.m@theagencygroup.co.nz',
                        'Carl Hastings 021 403 502 - mailto:carl@theagencygroup.co.nz',
                        'For more information or to view, please contact the agent."',
                        'Jeremy Langford on 021 278 0700 or email jeremy.langford@colliers.com',
                        'Not what you are after?',
                        'Mobile: 027 296 5989',
                        'For further information and to arrange an inspection call Terry on:',
                        'Key features of this property include:',
                        'For further information contact:',
                        'For more information contact:',
                        'For more information contact the master agents:',
                        'For more information please contact:',
                        'Call Luke Kershaw today to discuss - 021 610 093',
                        'For more information or to arrange an inspection contact:',
                        'SPACE DETAILS:',
                        'Contact details:',
                        'Mark Melville 022 154 6558 - mailto:mark.m@theagencygroup.co.nz or',
                        'Mark Melville (022 154 6558) - mailto:mark.m@theagencygroup.co.nz or',
                        'For further detail and to arrange an inspection call Terry on:',
                        'For more information and to arrange an inspection call Terry on:',
                        'For further information call Terry on:',
                        'Cell: 027 4411 015 or email: tom.burke@capitalrealty.co.nz',
                        '"Email:     terry@paulhastings.co.nz"""',
                        'Paul Soulis',
                        'Phone 027 4411 015 or email: tom.burke@capitalrealty.co.nz  Note: we have access to all listings – over 750  incl sole listings on the market.',
                        'Call Matthew today and let us help you find the perfect space for you!"',
                        'Key Features:',
                        'Price by negotiation.',
                        'Contact Tom Burke over 20 years’ experience & over 450 deals for full details of this listing & more private listings.',
                        'If you\'d like to see the full range of Wellington CBD or fringe spaces that meet your office relocation requirements give Luke a call today."',
                        '& more private listings.',
                        'Carl Hastings 021 403 502 - carl@the agencygroup.co.nz"',
                        'Carl Hastings (021 403 502) - carl@the agencygroup.co.nz',
                        'Quick Easy Search & up to date Office Market Report available at www.capitalrealty.co.nz"',
                        'Carl Hastings 021 403 502 - carl@the agencygroup.co.nz',
                        'For more information on this opportunity or any others please contact Jeremy Langford – Wellington Office Leasing specialist – 021-2780-700.',
                        'Steve Maitland on 021 726 200 or email steve.maitland@colliers.com',
                        "We'll be dealing direct with the owner who is very keen to do a deal!",
                        'Quick Easy Search & Market News available at www.capitalrealty.co.nz"',
                        'Property Details:',
                        'For more information about this space or to arrange a viewing, please contact:',
                        'Carl Hastings 021 403 502 - carl@theagencygroup.co.nz"',
                        'Email: terry@paulhastings.co.nz"',
                        '20 years’ experience and over 450 deals.',
                        'Office: 04 474 1585',
                        'OFFICE SPACE DETAILS:',
                        'Phone 027 4411 015 or email: tom.burke@capitalrealty.co.nz',
                        'Contact Tom Burke over 20 years’ experience & over 450 leasing deals for full details of this listing',
                        'Contact Tom Burke (over 20 years’ experience & over 450 leasing deals) for full details of this listing ',
                        '"Call Tom Burke on 027 4411015 to view"""',
                        '"Ring Tom Burke on 471 2364 now for full details."""',
                        '"Ring Tom Burke on 027 4411 015 to view."""',
                        'paul.soulis@cbre.co.nz"',
                        'Over 20 years’ experience and over 450 deals.',
                        'KEY FEATURES:',
                        'For more information about this space or other suitable space and to arrange a viewing, please contact:',
                        'Call Tom Burke now on 471 2364 to discuss this and other options in the building.',
                        'Contact Tom Burke for full details of this listing ',
                        'Call Tom Burke 027 4411 015 for full details',
                        '  Note: we have access to all listings (incl sole listings) on the market.',
                        '"For market news and more, go to www.capitalrealty.co.nz  ""',
                        '"Contact the leasing agent, Tom Burke (over 20 years’ experience & over 450 leasing deals) for full details of this listing and more private listings and to view this tenancy."',
                        'Email:   terry@paulhastings.co.nz"',
                        'Mobile:  027 296 5989',
                        'Email:    terry@paulhastings.co.nz"',
                        'See photos for floor plan',
                        'Over 20 years experience and over 450 deals.',
                        'Looking for cheap?',
                        'Key attributes:',
                        'For more information please contact:',
                        'For more information or to arrange an inspection contact Alastair Gustafson on 027 223 6013 or email mailto:al@agentcommercial.nz A member of the specialist CBD team.',
                        'Please contact Andrew Fullerton-Smith on 021 896 060 for more information.',
                        'Call Matt Clarke on 027 4409 608 to arrange inspection',
                        'Evan Price on 027 448 4199 or email mailto:evan.price@colliers.com',
                        'Steve Maitland on 021 726 200 or email mailto:steve.maitland@colliers.com',
                        'Jeremy Langford on 021 278 0700 or email mailto:jeremy.langford@colliers.com',
                        'More information: https://www.colliers.co.nz',
                        'Call Matthew today and let us help you find the perfect space for you!',
                        'For more information or to arrange an inspection contact Alastair Gustafson on 027 223 6013 or email mailto:al@agentcommercial.nz',
                        'For more information or to arrange an inspection contact Alastair Gustafson on 027 223 6013 or',
                        'COME VIEW TODAY For more information or to arrange an inspection contact Alastair Gustafson on 027 223 6013 or  A member of the specialist CBD team.',
                        'email mailto:al@agentcommercial.nz',
                        'cbre.co.nz',
                        "If you'd like to see the full range of Wellington CBD or fringe spaces that meet your office relocation requirements submit a brief at http://www.wellingtonofficespace.co.nz """
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
                      'provided','providing','may','needs','must','love','unbiased','size','help','negotiation','metrocommercial','www','http']
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

count_occur_df.to_csv('2_wlg_BagofWords.csv', encoding='utf-8', index=False)

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

vectors_df.to_csv('2_wlg_TF_iDF_keywords.csv', encoding='utf-8', index=False)

##----------------------------------------------Text Classification [Naive Bayes Classifier]----------------------------------------------
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




file.close()
print("2_wlg_vectorised run complete 100% +++++++++++++++++++++++++++++++++++++++")


