import csv
from sklearn.feature_extraction.text import CountVectorizer

##-----------------------------------------------open files-----------------------------------------------
filename = 'lemmatized_data.csv'  
with open (filename, encoding='utf-8') as lemmatized_text:
    csv_reader = csv.reader(lemmatized_text, delimiter=',')
    #data="".join(row[0] for row in csv_reader if isinstance(row[0], str))
    line_count = -1
    for row in csv_reader:
        if line_count == 0:
            print(f' {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[-1]}.')
            
    print(f'Processed {csv_reader.line_count} lines.')

'''
f = open(filename,"r",encoding="utf-8")
example_text = f.read()
example_text = example_text.replace(',', ' ')
print(example_text) 
'''

'''
count_vec = CountVectorizer()
count_occurs = count_vec.fit_transform([example_text])
print(count_occurs.toarray())


vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2,2))
print(count_vec.get_feature_names_out())

count_occur_df = pd.DataFrame(
    (count, word) for word, count in
    zip(count_occurs.toarray().tolist()[0],
    count_vec.get_feature_names_out()))
count_occur_df.columns = ['Word', 'Count']
count_occur_df.sort_values('Count', ascending=False,inplace=True)
count_occur_df.head()
print(count_occur_df.head())
print(count_vec.get_feature_names_out())

'''


