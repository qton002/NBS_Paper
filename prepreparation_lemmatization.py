import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

##-----------------------------------------------open files-----------------------------------------------
#use Excel, search & replace "grade a"/"a grade" to grade_a or a_grade; b_grade

filename = "test2.csv"  ##should use this csv file - data is tab:seismic_TRUE_noDuplicates - col:Ad_description. Copy&paste the column to create a new file using Excel and save as csv file.
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
 
####----------------------https://www.youtube.com/watch?v=hhjn4HVEdy0&list=PLP_4EPVEox99f1-_JMRpQbPkcGlMd5Xoq&index=3------------------------------------------------------------------------
##example_text = re.sub(r'[^\w\s]','',example_text) ##remove regression if ^not; \w word charaters; \s space cha
words = nltk.word_tokenize(no_punct_N_num) #tokenization
#print(words)

stop_words=stopwords.words('english')
additional_stopwds = ['youre', 'youve','youll','youd','shes','its','thatll','dont','shouldve','arent','couldnt',
                      'didnt','doesnt','hadnt','hasnt','havent','isnt','mightnt','mustnt','neednt','shant','shouldnt',
                      'wasnt','werent','wont','wouldnt','us','dont','would','sqm','sq','sqmtr','sqr','mtr','weve','theres',
                      'cant','th','sm','psm','whats','mtrs','thats','level','levels','meter','metre','sqmt','gst','pa','andor',
                      'however','today','call','available','ready','per','email','one','two','three','also','contact','approx',
                      'approximately','nla','gla','day','floor','floors','via','show','please','well','address','could',
                      'listing','id','still','touch','find','finding','found','able','st','yet','always','almost','although',
                      'among','anytime','call','rd']
stop_words.extend(additional_stopwds)
#print(stop_words)

words_no_stopwds= [w for w in words if not w.lower() in stop_words] #remove stopwords
#print(words_no_stopwds)

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in words_no_stopwds:
    lemmatized_words.append(lemmatizer.lemmatize(word)) ##lemmatizer words into meaningful dictionary form
#print(lemmatized_words)
    
filtered_sentence = ' '.join([str(item) for item in lemmatized_words])
print(filtered_sentence) ##copy & paste to Excel - tab:Lemmatization

f.close()

##-----------------------------------------------save cleaned data to a new csv file-----------------------------------------------
import csv

with open ('lemmatized_data.csv', 'w', encoding = 'UTF8') as newfile:
    writer = csv.writer(newfile)
    writer.writerow(lemmatized_words)


'''
##-----------------------------------------------remove stopwords-----------------------------------------------
from nltk.corpus import stopwords
stop_words=stopwords.words("english")
##print(stop_words)

##stop_words.append("add_new_word")
##stop_words.remove("remove_a_word")

word_tokenized = word_tokenize(no_punct)        ##remove punctuations
words_no_stopwds = []   ##create a new string

for w in word_tokenized:
    if w.lower() not in stop_words:
        words_no_stopwds.append(w)
print(words_no_stopwds)
'''
