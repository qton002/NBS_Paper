##-----------------------------------------------open files-----------------------------------------------
import pandas
df = pandas.read_csv('test2.csv')


##-----------------------------------------------remove punctuations-----------------------------------------------
punct = '''*-.,~\/&#+=;<>{}[]()?:@_$^|·•"'''    
##
no_punct = ""
for char in df:
    if(char not in punct):
        no_punct = no_punct + char
print(no_punct)
