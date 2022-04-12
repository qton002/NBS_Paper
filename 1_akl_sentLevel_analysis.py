import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
import nltk.corpus as Corpus
import numpy as np

file = pd.read_csv("1_akl_test2 - Copy.csv")
ad_descriptions = file.Ad_Desc.values
        
seismic_related_sents=[]
seismic_related_sents_array=[]

##_________________________________seismic-related sentences (start)_____________________________##
i=0
for sentences in ad_descriptions:
    sentences = sent_tokenize(ad_descriptions[i])
##    sentences= ' '.join([sents for sents in sentences])
##    sentences = re.sub(r'[\*\;]', '\n', sentences)
###    sentences= re.sub(r'[\-\–\·\•\…\_]', '\n', sentences) #removed '+' to show 'A+ Gradings'
##    #print(sentences)
    ##___
    sentence_arrays = [sentence.split('\n') for sentence in sentences]
    split_sentences = []
    for sub_list in sentence_arrays:
        for sentence in sub_list:
            sentence = sentence.strip()
            split_sentences.append(sentence.lower())
    split_sents= ' '.join([sents for sents in split_sentences])
    wordCountsEachSent = len(split_sents.split())
##    print("wordCountsEachSent",wordCountsEachSent)
    
    words = word_tokenize(split_sents) #tokenize words
    ps = PorterStemmer()
    ps_sents = ' '.join([ps.stem(w) for w in words])

##    test = ["nbs", "seismically", "iep", "earthquakes"]
##    for w in test:
##        print(w, " : ", ps.stem(w))
    
    ##___
    res = re.sub("[^\w]", " ",  ps_sents).split()
    for eachSent in split_sentences:
        if 'earthquake' in eachSent.lower(): ###################################################################################### word 'earthquake'
##            print(i," ",eachSent, "earthquake yes")
            res_earthquake = res.index('earthquak')+1
##            print(i," ","The location of 'earthquake' in sentence is : " + str(res_earthquake))
##            print(i," ","res_earthquake/wordCountsEachSent", res_earthquake/wordCountsEachSent)
            if (res_earthquake/wordCountsEachSent <= 1/3):
                earthquake_Pos = "Begining"
##                print(i," ","The location of 'earthquake' is at the Begining of the Ad_Desc")
            elif (res_earthquake/wordCountsEachSent > 1/3 and res_earthquake/wordCountsEachSent <= 2/3):
                earthquake_Pos = "Middle"
##                print(i," ","The location of 'earthquake' is in the Middle of the Ad_Desc")
            else:
                earthquake_Pos = "End"
##                print(i," ","The location of 'earthquake' is at the End of the Ad_Desc")
            seismic_related_sents.append(eachSent.strip())
            seismic_related_sents.append(earthquake_Pos)
            seismic_related_sents_array.append([i,eachSent.strip(), earthquake_Pos])
        elif 'iep' in eachSent.lower(): ########################################################################################################### word 'iep'
##            print(i," ",eachSent, "iep yes")
            res_iep = res.index('iep')+1
##            print(i," ","The location of 'iep' in sentence is : " + str(res_iep))
##            print(i," ","res_iep/wordCountsEachSent", res_iep/wordCountsEachSent)
            if (res_iep/wordCountsEachSent <= 1/3):
                iep_Pos = "Begining"
##                print(i," ","The location of 'iep' is at the Begining of the Ad_Desc")
            elif (res_iep/wordCountsEachSent > 1/3 and res_iep/wordCountsEachSent <= 2/3):
                iep_Pos = "Middle"
##                print(i," ","The location of 'iep' is in the Middle of the Ad_Desc")
            else:
                iep_Pos = "End"
##                print(i," ","The location of 'iep' is at the End of the Ad_Desc")
            seismic_related_sents.append(eachSent.strip())
            seismic_related_sents.append(iep_Pos)   
            seismic_related_sents_array.append([i,eachSent.strip(), iep_Pos])
        elif 'seismic' in eachSent.lower(): ########################################################################################################### word 'seismic'
##            print(i," ",eachSent, "seismic yes")
            res_seismic = res.index('seismic')+1
##            print(i," ","The location of 'seismic' in sentence is : " + str(res_seismic))
##            print(i," ","res_seismic/wordCountsEachSent", res_seismic/wordCountsEachSent)
            if (res_seismic/wordCountsEachSent <= 1/3):
                seismic_Pos = "Begining"
##                print(i," ","The location of 'seismic' is at the Begining of the Ad_Desc")
            elif (res_seismic/wordCountsEachSent > 1/3 and res_seismic/wordCountsEachSent <= 2/3):
                seismic_Pos = "Middle"
##                print(i," ","The location of 'seismic' is in the Middle of the Ad_Desc")
            else:
                seismic_Pos = "End"
##                print(i," ","The location of 'seismic' is at the End of the Ad_Desc")
            seismic_related_sents.append(eachSent.strip())
            seismic_related_sents.append(seismic_Pos)   
            seismic_related_sents_array.append([i,eachSent.strip(), seismic_Pos])
        elif 'nbs' in eachSent.lower(): ########################################################################################################### word 'nbs'
##            print(i," ",eachSent, "nbs yes")
            res_nbs = res.index('nb')+1
##            print(i," ","The location of 'nbs' in sentence is : " + str(res_nbs))
##            print(i," ","res_nbs/wordCountsEachSent", res_nbs/wordCountsEachSent)
            if (res_nbs/wordCountsEachSent <= 1/3):
                nbs_Pos = "Begining"
##                print(i," ","The location of 'nbs' is at the Begining of the Ad_Desc")
            elif (res_nbs/wordCountsEachSent > 1/3 and res_nbs/wordCountsEachSent <= 2/3):
                nbs_Pos = "Middle"
##                print(i," ","The location of 'nbs' is in the Middle of the Ad_Desc")
            else:
                nbs_Pos = "End"
##                print(i," ","The location of 'nbs' is at the End of the Ad_Desc")
            seismic_related_sents.append(eachSent.strip())
            seismic_related_sents.append(nbs_Pos)   
            seismic_related_sents_array.append([i,eachSent.strip(), nbs_Pos])
    i+=1
##print(seismic_related_sents_array)

seismic_related_SentsandPosition = pd.DataFrame(seismic_related_sents_array, columns = ['Sents_Index','Seismic_Related_Sentences', 'Word_Position'])
print(seismic_related_SentsandPosition)
seismic_related_SentsandPosition.to_csv('1_alk_seismic_related_SentsandPosition.csv', encoding='utf-8', index=False)

##vectors_df = pd.DataFrame(
##    (count, word) for word, count in
##    zip(wd_freq.tolist()[0],
##        feature_names_vectors))
##vectors_df.columns = ['Words','TF_iDF']
##vectors_df.sort_values('TF_iDF', ascending=False,inplace=True)
##
###print(vectors_df.head())
##print("vectors_df:", vectors_df)
##
##vectors_df.to_csv('1_akl_TF_iDF_keywords.csv', encoding='utf-8', index=False)

'''i=0
numOfAds=len(ad_descriptions)
print("number of ads in total", numOfAds)
#print("testing testing ad_descriptions[0]", ad_descriptions[0])
for i in range(2):
#for i in range(numOfAds):
    eachSent = tokenizer.tokenize(ad_descriptions[i])
    #eachSent = sent_tokenize(ad_descriptions[i])
    #print("eachSent", eachSent)
    numOfsentsinAd=len(eachSent)
    print("numOfsentsinAd", numOfsentsinAd)
    sents= ' '.join([sents for sents in eachSent])
    numOfwordsinAd = len(sents.split())
    print("numOfwordsinAd", numOfwordsinAd)
    current_Sent=eachSent[0].lower()
    print("current_Sent ", current_Sent)
    i+=1
print("seismic_related_sents", seismic_related_sents)

    for j in range(numOfsentsinAd):
        current_Sent=eachSent[j].lower()
        print("current_Sent[",i,j,"].lower(): ",current_Sent)
        print(current_Sent.index("earthquake"))
        try:
            if eachSent.find("earthquake"):            
                seismic_related_sents.append(i)
                seismic_related_sents.append(j)
                seismic_related_sents.append(eachSent[j].strip())
            j+=1
        except Exception:
            print("earthquake not in the sentence")'''
##    i+=1
##print("seismic_related_sents", seismic_related_sents)
#for s in seismic_related_sents:
  #  print(s)
##_________________________________seismic-related sentences (end)_____________________________##
'''
            res = re.sub("[^\w]", " ",  sents).split()
            earthquakeYN=None
            seismicYN=None
            nbsYN=None
            iepYN=None
        try:
            if eachSent[j].lower().index("earthquake") >1:
                res_earthquake = res.index("earthquake") + 1
                earthquakeYN=True
                print("The location of 'earthquake' in sentence", i,j, " is : " + str(res_earthquake))
        except Exception:
            print("The location of 'earthquake' in sentence has", i,j, "earthquake error")
        try:
            if eachSent[j].lower().index('seismic') >1:
                res_seismic = res.index('seismic') + 1
                seismicYN=True
                print("The location of 'seismic' in sentence", i,j, " is : " + str(res_seismic))
        except Exception:
            print("The location of 'seismic' in sentence has", i,j, "seismic error")
        try:
            if eachSent[j].lower().index('nbs') >1:
                res_nbs = res.index('nbs') + 1
                nbsYN = True
                print("The location of 'nbs' in sentence", i,j, "  is : " + str(res_nbs))
        except Exception:
            print("The location of 'nbs' in sentence has", i,j, "nbs error")
        try:
            if eachSent[j].lower().index('iep') >1:
                res_iep = res.index('iep') + 1
                iepYN = True
                print("The location of 'iep' in sentence", i,j, "  is : " + str(res_iep))
        except Exception:
            print("The location of 'iep' in sentence has", i,j, "iep error")'''


'''          wordindex = eachSent[j].lower().find('earthquake' or 'seismic' or 'nbs' or 'iep')
            #print("len(ad_descriptions[",i,j,"]",len(ad_descriptions[i]))
            if wordindex/len(ad_descriptions[i]) <= 1/3:
                seismic_related_sents.append("Pos_Begining")
            elif wordindex/len(ad_descriptions[i]) > 1/3 and wordindex/len(ad_descriptions[i]) <= 2/3:
                seismic_related_sents.append("Pos_Middle")
            else:
                seismic_related_sents.append("Pos_End")    '''

'''
        if earthquakeYN==True and seismicYN==True:
            print("both earthquakeYN and seismicYN is True")
        elif earthquakeYN==True and nbsYN==True:
            print("both earthquakeYN and nbsYN is True")
        elif earthquakeYN==True and iepYN==True:
            print("both earthquakeYN and iepYN is True")
        elif seismicYN==True and nbsYN==True:
            print("both seismicYN and nbsYN is True")
        elif seismicYN==True and iepYN==True:
            print("both seismicYN and iepYN is True")        
        elif seismicYN==True and iepYN==True:
            print("both nbsYN and iepYN is True")
        else:
            print("seismic related words only appeared once in ad_desc[",i,j,"]")

        try:
            if res_earthquake<res_seismic and res_nbs and res_iep:
                res = res_earthquake
                print("The location of 'res_earthquake' in sentence", i,j, "  is : " + str(res_earthquake))
            elif res_seismic<res_earthquake and res_nbs and res_iep:
                print("The location of 'res_seismic' in sentence", i,j, "  is : " + str(res_seismic))                
            elif res_nbs<res_earthquake and res_seismic and res_iep:
                print("The location of 'res_nbs' in sentence", i,j, "  is : " + str(res_nbs))
            elif res_iep<res_earthquake and res_seismic and res_nbs:
                print("The location of 'res_iep' in sentence", i,j, "  is : " + str(res_iep))
            else:
                print("no seismic related words exists in the sentence",i,j)
        except:
            pass'''
##        else:
##            print ("This sentence does not contain a seismic-related word.")
##            res = re.sub("[^\w]", " ",  sents).split()
##            res = res.index(wrd) + 1
##            # printing result 
##            print("The location of word is : " + str(res)) 



#numOfsentsinAd=len(ad_descriptions[i])-1
#sents= ' '.join([sents for sents in ad_descriptions[i]])       
#numOfwordsinAd = len(sents.split())

eachAd_index= 0
'''for eachAd_index in range(len(ad_descriptions)-1):
    len_eachAd = len(ad_descriptions[eachAd_index].split())-1 #word count for each ad
    i=0
    for i in range(len_eachAd):   
        eachSent = ad_descriptions[eachAd_index][i]
        numOfwords = len(eachSent.split())-1  #word count for each sentence in the ad
        print("numOfwords for[ad_descriptions[",eachAd_index,"]", numOfwords)

        if "earthquake" in eachSent.lower() or "seismic" in eachSent.lower() or "nbs" in eachSent.lower() or "iep" in eachSent.lower():
            new_testing0.append(eachSent.strip())
            print("testing0[",i,"].strip()",eachSent.strip())
            i+=1
        else:
            print("no seismic related sentences.")
            break
    eachAd_index +=1'''
        
##for i in testing:
##    print("testing", i)

##tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
##for eachSentence in ad_descriptions:
##    #eachSentence = re.sub(r'[\*\;]', '.', eachSentence)
##    #eachSentence = re.sub(r'[\-\–\·\•\…\_\'\"\’\”\‘\“]', ' ', eachSentence) #removed '+' to show 'A+ Gradings'
##    sent_text = tokenizer.tokenize(eachSentence)
##    print("sent_text", sent_text)

##_________________________________getting unique sentences (start)_____________________________##
filename = "1_akl_test2.csv"  
f = open(filename,"r",encoding="utf-8")
ad_descriptions = f.read()

ad_descriptions = re.sub(r'[\*\;]', '\n', ad_descriptions)
ad_descriptions = re.sub(r'[\-\–\·\•\…\_\'\"\’\”\‘\“]', ' ', ad_descriptions) #removed '+' to show 'A+ Gradings'

sent_text = sent_tokenize(ad_descriptions)
sentences_arrays = [sentence.split('\n') for sentence in sent_text]
sentences = []
for sub_list in sentences_arrays:
    for sentence in sub_list:
        sentence = sentence.strip()
        sentences.append(sentence.lower())
#print("sentences: ", sentences)
##for i in sentences:
##   print(i)
f.close()
 
clean_sentences=[]
##stop_words_remove = ['a', 'an','b','c']
##stop_words = set(stopwords.words('english')).difference(stop_words_remove)
for sents in sentences:
    #filtered_sentences = [w for w in word_tokenize(sents) if not w.lower() in stop_words]
    sents = re.sub(r'[\.\?]', '', sents) #remove '.' to no space
    filtered_sentences = [w for w in word_tokenize(sents)]
    filtered_sentences = ' '.join([sents for sents in filtered_sentences])
    clean_sentences.append(filtered_sentences.strip())
#print("clean_sentences:", clean_sentences)
#print("len(clean_sentences)-1",len(clean_sentences)-1)

##for i in clean_sentences:
##    print(i)

'''#seismic_sents = pd.read_csv("1_akl_sent_text_nodupli_seismic_only.csv")
seismic_sents = pd.read_csv("1_akl_sent_text_nodupli.csv")   
data = seismic_sents.seismic_sents.values
clean_sents=[]
#stop_words_remove = ['a', 'an','b','c']
#stop_words = set(stopwords.words('english')).difference(stop_words_remove)
for sents in data:
    #filtered_sents = [w for w in word_tokenize(sents) if not w.lower() in stop_words]
    filtered_sents = [w for w in word_tokenize(sents)]
    filtered_sents = ' '.join([sents for sents in filtered_sents])
    clean_sents.append(filtered_sents.strip())
#print("clean_sents:", clean_sents)'''

seismic_yes_only = []
for i in clean_sentences:
    if "earthquake" in i or "seismic" in i or "nbs" in i or "iep" in i:
        if not i[0] == "+":
            i = i.strip()
            seismic_yes_only.append(i)
        #print(i.strip())

vocab=nltk.FreqDist(seismic_yes_only)
#print("Length of vocab: ", len(vocab))
freq_tuple=vocab.most_common(50)
freq_array = np.asarray(freq_tuple)
#print("the most freq sents:", freq_array)

seismic_yes_noDups=list(dict.fromkeys(seismic_yes_only))
seismic_yes_noDups.sort()

#for i in seismic_yes_noDups:
    #print(i)
    

'''icontainsi = []
i=0
icontainsi.append(seismic_yes_noDups[i])
for i in range(len(seismic_yes_noDups)-1):
    icontainsi.append(seismic_yes_noDups[i+1])
    if re.search(seismic_yes_noDups[i],seismic_yes_noDups[i+1]):
        #print("found")
        icontainsi.remove(seismic_yes_noDups[i])
    elif re.search(seismic_yes_noDups[i+1],seismic_yes_noDups[i]):
        #print("found elif")
        icontainsi.remove(seismic_yes_noDups[i+1])
    i+=1
icontainsi.sort()
##for i in icontainsi:
##    print(i)'''

'''
phrase_freq = []
counter = 0
i=0
for i in range (len(icontainsi)-1):
    j = 0
    for j in range(len(clean_sentences)-1):
        if icontainsi[i] in clean_sentences[j]:
            counter +=1
            #phrase_freq.append()
            j += 1
        else:
            counter = counter
            j+=1
    i+=1
    #print(icontainsi[i], ": ", counter)

##_________________________________getting unique sentences (end)_____________________________##

lemmatizer = WordNetLemmatizer()
icontainsi_lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in icontainsi])

from sklearn.feature_extraction.text import CountVectorizer
pd.set_option("display.max_rows", None, "display.max_columns", None) ##set this to display all results
count_vec = CountVectorizer(analyzer='word', ngram_range=(3,5))
count_occurs = count_vec.fit_transform([icontainsi_lemmatized])
feature_names = count_vec.get_feature_names() ##in array
feature_names_txt ='\n'.join([str(item) for item in feature_names])  ##in str
#print(feature_names_txt2)

count_occur_df = pd.DataFrame(
    (count, word) for word, count in
    zip(count_occurs.toarray().tolist()[0],
    count_vec.get_feature_names_out()))
count_occur_df.columns = ['Word2','Word_Freq']
count_occur_df.sort_values('Word_Freq', ascending=False,inplace=True)

print("count_occur_df: ",count_occur_df)
#print(count_occur_df2.head())

'''

'''
filename = "1_akl_test2.csv"  
f = open(filename,"r",encoding="utf-8")
ad_descriptions = f.read()

##_________________________________sentences level analysis (OPEN)_____________________________##
ad_descriptions = re.sub(r'[\*\;]', '\n', ad_descriptions)
ad_descriptions = re.sub(r'[\-\–\+\·\•\…\_\'\"\’\”\‘\“]', ' ', ad_descriptions)
sent_text = sent_tokenize(ad_descriptions)
sentences_arrays = [sentence.split('\n') for sentence in sent_text]
sentences = []
for sub_list in sentences_arrays:
    for sentence in sub_list:
        sentences.append(sentence)
#print("sentences: ", sentences)

####RUN ONCE ONLY!!!!!!!!!!!!.################################################
##df = pd.DataFrame(sentences)
##df.columns = ["Ad_Desc"]
##df.sort_values('Ad_Desc', ascending=False,inplace=True)
##df .to_csv('1_akl_sent_text.csv', encoding='utf-8', index = False, header=False) #save to new file. Manully classify the words.
##________________________________
'''
'''vocab=nltk.FreqDist(sentences)
print("Length of vocab: ", len(vocab))
freq_tuple=vocab.most_common(200)
freq_array = np.asarray(freq_tuple)
print("the most freq words (200) freq_array:", freq_array) #help to remove agency_related contents
#print("vocab.most_common(200):", vocab.most_common(200)) #help to remove agency_related contents

df_most_common = pd.DataFrame(freq_array)
df_most_common.columns=["most_freq","freq"]
df_most_common.to_csv('1_akl_most_common200.csv', encoding='utf-8')

####RUN ONCE ONLY!!!!!!!!!!!!.################################################

##df = pd.DataFrame(list(set(sentences)))
##df.columns = ["Ad_Desc"]
##df.sort_values('Ad_Desc', ascending=False,inplace=True)
##df .to_csv('1_akl_sent_text_nodupli.csv', encoding='utf-8', index = False, header=False) #save to new file. Manully classify the words.
##_____________________________________________________________________________________________________________________________
## In EXCEL - 1.rep'.!'to' ' & rep'~?'to' '; 2. trim(); 3.LowerCase; 4. remove duplicates; 5. remove blank rows; 6.filter Earthquake-related only (Seismic_YorN).
## copy only seismic_YorN = TRUE to a new file = 1_akl_sent_text_nodupli_seismic_only.csv
##_____________________________________________________________________________________________________________________________


##_________________________________sentences level analysis (CLOSE)_____________________________##
####_____________________________________________________________________________________####

f.close()
'''
'''
###____________________________________________________####
filename_sent_text = "1_akl_sent_text.csv"  
with open(filename_sent_text,"r",encoding="utf-8") as file:
    mystring = file.readlines()
    for i,line in enumerate(mystring):
        for pattern in ["If you are reviewing your office needs, we can assist.",
                        "Metro is an independent property agency offering you unbiased advice.",
                        "metrocommercial.co.nz",
                        "metrocommercial",
                        "Bayley's Commercial provides access to every leasing opportunity in Auckland.",
                        "Contact one of our workplace specialists for a bespoke and thorough market search.",
                        "At Frank we take a proactive approach with tenants and by understanding your needs provide a range of suitable commercial property options.",
                        "We have many more options available so if you can't find what you're looking for feel free to give us a call.",
                        "We are actively speaking with landlords of all property types and sizes and would love to help you find your perfect space.",
                        "We’re actively speaking with landlords of all property types and sizes, and would love to help you find the perfect space for your business.",
                        "Floor plans are available on request.",
                        "Contact us to experience a refreshing approach to commercial real estate.",
                        "We have more options available and many of these are off-market so to receive your custom list with full pricing information call us.",
                        "If you can't find what you're looking for or don't have the time email us your requirement and we'll keep you updated with potentially suitable options.",
                        "Can't find the right space for you?",
                        "Connect with us:",
                        "Facebook: facebook.com/frank commercial",
                        "Key features:",
                        "Instagram: instagram.com/frank commercial",
                        "Please don't hesitate to call our agents regarding 87 Albert Street or to inform us of your requirements.",
                        "please don’t hesitate to call or email one of the team on the details listed below.",
                        "Metro Commercial bring a point of difference in Commercial Real Estate as an independent agency offering unbiased advice.",
                        "We have 100's of listings across all categories and we would enjoy the opportunity to work with you to find the property solution best suited for you and your business.",
                        "We have 100's of listings across all categories, and we would enjoy the opportunity to work with you to find the property solution best suited for you and your business.",
                        "KEY FEATURES",
                        "Call Michaela 027 532 7333 or Neelam 021 885 915.",
                        "LinkedIn: nz.linkedin.com/company/frank commercial",
                        " LinkedIn: nz.linkedin.com/company/frank commercial",
                        "Show more",
                        "Call now to arrange an inspection.",
                        "Connect with us!",
                        "For more information visit:",
                        "The perfect office space for you is only a phone call away.",
                        "Our dedicated team of property professionals can help get your business moving and through their connections and expertise will have you in the right space in no time.",
                        "Contact us to experience a refreshing approach to commercial real estate.",
                        "Key Features: ",
                        "Contact us today on 09 366 1666!",
                        "Brendan Graves",
                        "We look forward to hearing from you soon.",
                        "www.officefocus.co.nz",
                        "Whether your company is expanding contracting or even thinking about renegotiating/ renewing your existing lease - call the Colliers International office specialists on 09 358 1888 or email us at auckland.leasing@colliers.com",
                        "Whether your company is expanding, contracting or even thinking about renegotiating/ renewing your existing lease - call the Colliers International office specialists on 09 358 1888 or email us at auckland.leasing@colliers.com ",
                        "Colliers International look forward to assisting your requirement.",
                        "Colliers look forward to assisting with your requirement.",
                        "For the latest market insights and high quality Auckland office options go to Colliers LEASE 2021 - https://officelease.colliers.co.nz/",
                        "Call the Colliers International office specialists on 09 358 1888 or email us at auckland.leasing@colliers.com",
                        "We currently have over 1,700 spaces available in the Auckland CBD and city fringe.",
                        "Contact us for help finding the right environment for you and your business.",
                        "Facebook: facebook.com/CBRE",
                        "Twitter: twitter.com/cbreNewZealand",
                        "Google +: plus.google.com/+CBRE/posts",
                        "We currently have a range of properties available throughout Auckland",
                        "For more information or to arrange a viewing call or email City Fringe Sales & Leasing specialists",
                        "At Frank we take a proactive approach with tenants, and by understanding your needs provide a range of suitable commercial property options.",
                        "More information: https://www.colliers.co.nz",
                        "Connect with us: Facebook: facebook.com/frank commercial Instagram: instagram.com/frank commercial",
                        "call today to","For more information and to arrange an inspection",
                        "Facebook: facebook.com/CBRE Twitter: twitter.com/cbreNewZealand  Show more",
                        "Facebook: facebook.com/CBRE Twitter: twitter.com/cbreNewZealand  Show more""",
                        "For further information or to arrange an inspection contact:",
                        "@cbre.co.nz",
                        "co.nz",
                        "Rent by negotiation",
                        "Rents are by negotiation",
                        "Rents by negotiation.",
                        "We have many more options available, so if you can't find what you're looking for feel free to give us a call.",
                        "We have more options available and many of these are off-market, so to receive your custom list with full pricing information call us.",
                        "We have plenty more office options available and many of these are off-market, so to receive your custom list with full pricing information call us.",
                        "If you can't find what you're looking for or don't have the time, email us your requirement and we'll keep you updated with potentially suitable options.",
                        "If you can't find what you're looking for or don't have the time, email us your requirement and we'll keep you updated with potentially suitable options before they come online.",
                        "For a full range of currently available properties check out www.jamesgroup.co.nz ",
                        "jamesgroup.co.nz",
                        "For more information and to inspect please contact us."
                        ]:
            if pattern in line:
                mystring[i] = line.replace(pattern,"")                    
    # print the processed lines
    remove_agent_related_info = "".join(mystring)
    print("remove_agent_related_info: ",remove_agent_related_info) #agency_related contents removed as much as possible
    
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

print("count_occur_df: ",count_occur_df)
#print(count_occur_df2.head())

count_occur_df.to_csv('1_akl_BagofWords.csv', encoding='utf-8', index=False)

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
print("vectors_df:", vectors_df)

vectors_df.to_csv('1_akl_TF_iDF_keywords.csv', encoding='utf-8', index=False)

##----------------------------------------------Text Classification [Naive Bayes Classifier]----------------------------------------------
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



file.close()
'''
print("1_akl_vectorised run completed 100%++++++++++++++++++++++++++++++++++++")

