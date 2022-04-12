import pandas as pd
data = pd.read_csv("test2 - Copy.csv")

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tqdm import tqdm
def data_cleaner (data):
    clean_data = []
    for sentence in tqdm(data):
        cleantext = re.sub(r'[^\w\s]', ' ', sentence)
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
        filtered_sentence = [w for w in word_tokenize(cleantext) if not w.lower() in stop_words]

        lemmatizer = WordNetLemmatizer()
        cleantext = ' '.join([lemmatizer.lemmatize(w) for w in filtered_sentence if w.isalpha()])

        clean_data.append(cleantext.strip())
    return clean_data

clean_data = data_cleaner(data.Ad_Description.values)  #called the function def data_cleaner (data):
#print(clean_data[0])


##----------------------------------------------TF-iDF----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option("display.max_rows", None, "display.max_columns", None)  ##set this to display all results

TfidfVectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2,3))
vectors = TfidfVectorizer.fit_transform(clean_data)

wd_freq = vectors.toarray()
feature_names_vectors = TfidfVectorizer.get_feature_names() ##in array

vectors_df = pd.DataFrame(
    (count, word) for word, count in
    zip(wd_freq.tolist()[0],
        feature_names_vectors))
vectors_df.columns = ['Words','TF_iDF']
vectors_df.sort_values('TF_iDF', ascending=False,inplace=True)
print(vectors_df)
       
'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, data.Ad_Description, test_size=0.2, random_state=42,stratify=data.Ad_Description)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)'''
        


