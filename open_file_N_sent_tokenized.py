##-----------------------------------------------open files-----------------------------------------------
filename = "test2.csv"  ####should use this csv file - data is tab:seismic_TRUE_noDuplicates - col:Ad_description. Copy&paste the column to create a new file using Excel and save as csv file.
f = open(filename,"r",encoding="utf-8")
example_text = f.read()
example_text = example_text.replace("\t", " ")

##-----------------------------------------------remove punctuations-----------------------------------------------
punct = '''*-.,~\/&#+=;<>'’‘{}[]()?:@_$^—|·•"–“”…!'''
##
no_punct = ""
for char in example_text:
    if(char not in punct):
        no_punct = no_punct + char

##-----------------------------------------------sent_tokenize-----------------------------------------------
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sent_tokenized = ""
for i in sent_tokenize(no_punct):
    sent_tokenized = sent_tokenized + i + "\n"

print(sent_tokenized)


'''
1. copy & paste result to Excel (in tab:NLTK_no_punc_sent_token), and manually remove whitespaces
and align data into coloum A;
2. TRIM()
3. create a new column "Seismic_YES" use formula
=OR(ISNUMBER(SEARCH("nbs",B2))=TRUE,ISNUMBER(SEARCH("seismic",B2))=TRUE,ISNUMBER(SEARCH("earthquake",B2))=TRUE,ISNUMBER(SEARCH("IEP",B2))=TRUE)
4. new column for selected_seismic_true_only
5. use Remove Duplicates functions under the DATA tab in Excel to remove duplicates
6. Analyse from here. Clean and make meaningful intepretations.
6.1. copy & paste print(sent_tokenized) to a new file: sent_tokenized.txt for Bag_of_words 
6.2. copy & paste coloum 'Selected_True_ONLY' to a new file: sent_tokenized_true_only.txt for Bag_of_words
'''



















