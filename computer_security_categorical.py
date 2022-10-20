import os
import re
import pandas as pd
import pickle 
import statistics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import pandas as pd
from sklearn.metrics import accuracy_score
from itertools import cycle
import sklearn.neighbors
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import sklearn.linear_model
from collections import Counter
import os
!pip uninstall h5py
import tensorflow as tf
!pip install tensorflow==2.1.0
tf.test.gpu_device_name
assert()
tf.config.experimental.list_physical_devices('GPU')
tf.__version__
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from sklearn.utils import shuffle
from sklearn.svm import SVC
!pip install scipy
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn import metrics
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
optimizer = optimizers.Adam(clipvalue=0.001)
import re
if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")
num_pipeline = Pipeline([
        
        ('std_scaler', StandardScaler())
    ])
num_pipelines = Pipeline([
        
        ('min_scaler', MinMaxScaler())
    ]) # neural networks dataset


path = "C:\\Users\\admin\\Documents\\apiecate"

path = "C:\\Users\\admin\\Documents\\apie"

path = "C:\\Users\\admin\\Documents\\sccate"

path = "C:\\Users\\admin\\Documents\\scie"

path = "D:\\malware-classification\\train_run1"

path = "D:\\malware-classification\\test_run1"

lis =[]

def openFile(fileName):
    fileHere = os.path.join(path,fileName)
    with open(fileHere, 'r' , encoding='utf-8') as f:
        lis.append( f.read().splitlines())

lis[1342]        


files = os.listdir(path)
for file in files:
    openFile(file)
    
    

uniques = set();
for file in lis:
    for val in file:
        uniques.add(val)
        
uniques = list(uniques)

listest =[]
listest.append(lis[0])
listest.append(lis[1])

dictHere = dict()

for unq in uniques:
    dictHere[unq]=dict()
    for fileno,file in enumerate(lis):
        dictHere[unq][ files[fileno] ]=[]
        for idx ,val in enumerate(file):
            if(val ==unq):
                
                dictHere[unq][ files[fileno] ].append(idx)
                
                
df = pd.DataFrame(dictHere)



df.to_pickle('df.pkl')

df.to_csv('apiefinal_txt.csv' )

new_df = pd.DataFrame()



df = pd.read_csv("C:\\Users\\admin\\Documents\\Scie_output\\apie_txt.csv")

col = list(df.columns)
new_df['Text']= col

for unq in col:
    lisre = list(df[unq])
 
    lisFinal =[]
    for row in lisre:
        row = list(np.diff(row))
        if(row==[]):
            lisFinal.append([])
      
        else:  
              
            maxHere = max(row)
            minHere = min(row)
            avgHere = statistics.mean(row)
            medianhere = statistics.median(row)
            lisFinal.append([minHere,avgHere,maxHere,medianhere])
    new_df[unq]=lisFinal

df.to_pickle('new_df.pkl')

minlen=[]
meanlen=[]
maxlen=[]
medianlen=[]

for j in range(0,len(new_df)):
    print(j)
    for k in range(0,len(new_df.iloc[j])):
        print(k)
        if(len(new_df.iloc[j][k])!=0):
            medianlen.append(new_df.iloc[j][k][0])

meanlenlength = np.mean(medianlen)


def my_cool_preprocessor(text):
    
    text=text.lower() 
    text=re.sub("\\W"," ",text) # remove special chars
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words
    
    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)

######   without category
# minlengthapi = 1347
# medianlengthapi = 3952
# meanlengthapi = 4409
# maxlengthapi = 6919
# medianlengthsc = 18
# maxlengthsc = 363
# meanlengthsc = 79
# minlengthsc = 5

##################   with category

# minlengthapi = 297
# meanlengthapi = 729
# maxlengthapi = 3717
# medianlengthapi = 461
# minlengthsc = 3
# avglenghthsc = 13
# maxlengthsc = 2223
# medianlensc = 3

##################   with category microsoft
# medianlengthapi = 16
# maxlengthapi = 37
# meanlengthapi = 18
# minlengthapi = 11













# create dataset min length sys call (wcategory)

c_vec = CountVectorizer(ngram_range=(4,4))
mxt_rep=[]


def generate_N_grams(text,ngram):
  words=[word for word in text.split(" ")]  
  print("Sentence after removing stopwords:",words)
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans




def openFilesng(fileName):
    
    fileHere = os.path.join(path,fileName)
    file = open(fileHere,'r' , encoding='utf-8')
    
    line = file.read().replace("\n", " ")
    file.close()
    asn = generate_N_grams(line,4)   
    xcv = set(asn)
    pkt = list(xcv)
    

def openFiles(fileName):
    
    
    lis.clear()
    fileHere = os.path.join(path,fileName)
    with open(fileHere, 'r' , encoding='utf-8') as f:
        lis.append( f.read().splitlines())
      
    xcv = np.array(lis[0])
    pkt = np.unique(xcv)
    # if(len(lis[0])%2!=0):
    #     del(lis[0][len(lis[0])-1])
    ## replace value after  / to sliding window length    
    listlengthcal = len(lis[0]) / 37
    xmas = ' '.join(map(str,lis[0]))
    if(listlengthcal>=1):
        
        n_split = [(lis[0][i:i+37]) for i in range(0,len(lis[0]),37)]
       
        for split in n_split:
            
            ngrams = c_vec.fit_transform(xmas.split('\n'))
         
            vocab = c_vec.vocabulary_
            count_values  = ngrams.toarray().sum(axis=0)
            quartile=pd.Series(list(count_values)).quantile([0.25,0.75])
            quart = list(quartile)
            mxt  = [(count_values[i],k) for k,i in vocab.items()]
            for k in mxt:
                
                if quart[0] <= k[0] <= quart[1]:
                    mxt_rep.append(k[1])
                    
    elif(0<listlengthcal<1):
         
         ngrams = c_vec.fit_transform(xmas.split('\n'))
         vocab = c_vec.vocabulary_
         count_values  = ngrams.toarray().sum(axis=0)
         quartile=pd.Series(list(count_values)).quantile([0.25,0.75])
         quart = list(quartile)
         mxt  = [(count_values[i],k) for k,i in vocab.items()]
         for k in mxt:
            if quart[0] <= k[0] <= quart[1]:
                mxt_rep.append(k[1])
              
   
       
        
        
files = os.listdir(path)
countersp =0
for file in files:
    print(file)
    print(countersp)
    openFiles(file)
    countersp = countersp+1

for k in range(7380,len(files)):
    print(files[k])
    print(countersp)
    openFiles(files[k])
    countersp = countersp+1
    

files[6500]
mxt_rep = set(mxt_rep)
mxt_rep = list(mxt_rep)

c_vecp = CountVectorizer(ngram_range=(4,4))
df3=[]
df2=[]
label_f=[]
for file in files:
    print(file)
    file = file.lower()
    # r = re.compile("([a-zA-Z]+)([0-9]+)")
    # match = r.match(file)
    file_split = file.split("_")
    # match = re.match(r"([a-z])",file_split[0])
    # items = match.groups()
    # items[0]
    df3.clear()
    lis.clear()
    fileHere = os.path.join(path,file)
    with open(fileHere, 'r' , encoding='utf-8') as f:
        lis.append( f.read().splitlines())
    if(len(lis[0])>0):
        #label_f.append(file_split[0])
        label_f.append(file_split[0])
        xmas = ' '.join(map(str,lis[0]))
        ngrams = c_vecp.fit_transform(xmas.split('\n'))
        
        vocab = c_vecp.vocabulary_
        count_values  = ngrams.toarray().sum(axis=0)
        mxt  = [(count_values[i],k) for k,i in vocab.items()]
        mxt_val = list(vocab.keys())
        for z in range(0,len(mxt_rep)):
           
           
            if(mxt_rep[z] in mxt_val):
                indc = mxt_val.index(mxt_rep[z])
                df3.append(mxt[indc][0])
            else:
                df3.append(0)
        df2.append(df3[:])
        
    

df3 = pd.DataFrame(df2, columns=mxt_rep)
df3['label']=label_f

df3.to_csv("apicmax4gram+microsoft+final+test.csv")

# train a model
oecd_bli = pd.read_csv("C:\\Users\\admin\\Documents\\Scie_output\\scc_min_3gram.csv")
oecd_bli.drop(oecd_bli[oecd_bli['label']=="scc"].index,inplace = True)
ulabel = oecd_bli["label"].copy()
ulabel_codes = pd.Categorical(ulabel).codes
df  = shuffle(oecd_bli)
Counter(df['label'])
split = StratifiedShuffleSplit(n_splits=2, test_size=0.33, random_state=42)
for train_index, test_index in split.split(df,df['label']):
    strat_train_set = df.iloc[train_index]
    strat_test_set = df.iloc[test_index]  

housing = strat_train_set.copy()
housing = strat_train_set.drop("label", axis=1) 
housing_labels = strat_train_set["label"].copy()
housing_labels_series = strat_train_set["label"].copy()
X_test = strat_test_set.drop("label", axis=1)
y_test = strat_test_set["label"].copy()
y_test_series = strat_test_set["label"].copy()
housing_labels = pd.Categorical(housing_labels).codes
# housing_labels = pd.factorize(housing_labels)[0]
y_test = pd.Categorical(y_test).codes
# y_test = y_test.astype('category').cat.codes
# housing_prepared_test = num_pipelines.fit_transform(X_test)
# housing_prepared = num_pipelines.fit_transform(housing)
svm_clf=SVC()
housing = num_pipelines.fit_transform(housing)# neural networks
X_test = num_pipelines.fit_transform(X_test)



classifier = OneVsRestClassifier(
    svm.SVC(kernel="linear", probability=True, random_state= np.random.RandomState(0))
)
y_score = classifier.fit(housing, housing_labels).decision_function(X_test)
housing_labels = label_binarize(housing_labels, classes=[0, 1, 2,3,4,5])
y_test = label_binarize(y_test, classes=[0, 1, 2,3,4,5])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['blue', 'red', 'green','yellow','black','orange'])
for i, color in zip(range(6), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

file1 = open('myfile.txt', 'w')
file1.writelines(L)
file1.close()
 
# Using readlines()
file1 = open('C:\\Users\\admin\\Documents\\Scie_output\\malware_api_class-master\\malware_api_class-master\\mal-api-2019\\all_analysis_data.txt', 'r')
Lines = file1.readlines()
 
i = 0

for line in Lines:
   
    if(labelcsv[i]=='Backdoor' or labelcsv[i]=='Dropper' or labelcsv[i]=='Virus' or labelcsv[i]=='Trojan' or labelcsv[i]=='Worms'):
        csav = path + labelcsv[i] + str(i)+ ".txt"
        f = open(csav,'w')
        lispie =line.split(" ")
      
 

        for ele in lispie:
            f.write(ele+'\n')
        f.close() 
    i=i+1
  
# reading apiindex file
file1 = open('C:\\Users\\admin\\Documents\\Scie_output\\malware_api_class-master\\malware_api_class-master\\ApiIndex.txt', 'r')
Lines = file1.readlines()
dict_api ={}

for line in Lines:
    lispie = line.split("=")
    dict_api[int(lispie[1])] = lispie[0]
    
dict_api[111]="111"

dict_api[109]="109"

for k in range(0,len(beniapi)):
    if(beniapi.loc[k,'malware']==0):
        lsyup = list(beniapi.iloc[k])
        del(lsyup[0])
        del(lsyup[len(lsyup)-1])
        for l in range(0,len(lsyup)):
            lsyup[l] = dict_api[lsyup[l]]
        csav = path + "benign" + str(k)+ ".txt"
        f = open(csav,'w')

        for ele in lsyup:
            f.write(ele+'\n')
        f.close() 

# api category dictionary
path = "C:\\Users\\admin\\Documents\\apiecate\\"

ApiIndexcsv =pd.DataFrame(ApiIndexcsv)

liscate =[]
combilis=[]

for p in range(0,len(ApiIndexcsv)):
    combilis.append(ApiIndexcsv.iloc[p][0])
    combilis.append(ApiIndexcsv.iloc[p][1])

res_dc = {combilis[i]:combilis[i+1] for i in range(0,len(combilis),2)}

def openFilecate(fileName):
    liscate.clear()
    fileHere = os.path.join(path,fileName)
    with open(fileHere, 'r' , encoding='utf-8') as f:
        liscate.append( f.read().splitlines())
    C = (pd.Series(liscate[0])).map(res_dc)
    D =list(C)
    with open(fileHere, 'w') as f:
        for item in D:
            f.write("%s\n" % item)
            
            
files = os.listdir(path)
for file in files:
    openFilecate(file)
                
# writing api calls
fileapitxt = "C:\\Users\\admin\\Documents\\Scie_output\\malware_api_class-master\\malware_api_class-master\\mal-api-2019\\all_analysis_data.txt"
lopendra=[]
with open(fileapitxt,'r',encoding='utf-8') as f:
    lopendra.append( f.read().splitlines())
    
    
    for item in D:
        f.write("%s\n" % item)    
# sc category dictionary
path = "C:\\Users\\admin\\Documents\\scie\\"

ApiIndexcsv =pd.DataFrame(scindexcsv)

liscate =[]
combilis=[]

for p in range(0,len(ApiIndexcsv)):
    combilis.append(ApiIndexcsv.iloc[p][0])
    combilis.append(ApiIndexcsv.iloc[p][1])

res_dc = {combilis[i]:combilis[i+1] for i in range(0,len(combilis),2)}

def openFilecate(fileName):
    liscate.clear()
    fileHere = os.path.join(path,fileName)
    fileh = os.path.join("C:\\Users\\admin\\Documents\\sccate",fileName)
    with open(fileHere, 'r' , encoding='utf-8') as f:
        liscate.append( f.read().splitlines())
    C = (pd.Series(liscate[0])).map(res_dc)
    D =list(C)
    with open(fileh, 'w') as f:
        for item in D:
            f.write("%s\n" % item)
            
            
files = os.listdir("C:\\Users\\admin\\Documents\\scie")
for file in files:
    openFilecate(file)
                
# writing api calls
fileapitxt = "C:\\Users\\admin\\Documents\\Scie_output\\malware_api_class-master\\malware_api_class-master\\mal-api-2019\\all_analysis_data.txt"
lopendra=[]
with open(fileapitxt,'r',encoding='utf-8') as f:
    lopendra.append( f.read().splitlines())
    
    
    for item in D:
        f.write("%s\n" % item)      
        

# api calls category roc curve
oecd_bli = pd.read_csv(r"C:\Users\admin\apicmin1gram.csv")
ulabel = oecd_bli["label"].copy()
ulabel_codes = pd.Categorical(ulabel).codes
df  = shuffle(oecd_bli)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df,df['label']):
    strat_train_set = df.iloc[train_index]
    strat_test_set = df.iloc[test_index]  

housing = strat_train_set.copy()
housing = strat_train_set.drop("label", axis=1) 
housing_labels = strat_train_set["label"].copy()
housing_labels_series = strat_train_set["label"].copy()
X_test = strat_test_set.drop("label", axis=1)
y_test = strat_test_set["label"].copy()
y_test_series = strat_test_set["label"].copy()
housing_labels = pd.Categorical(housing_labels).codes
# housing_labels = pd.factorize(housing_labels)[0]
y_test = pd.Categorical(y_test).codes
# y_test = y_test.astype('category').cat.codes
# housing_prepared_test = num_pipelines.fit_transform(X_test)
# housing_prepared = num_pipelines.fit_transform(housing)
svm_clf=SVC()
housing = num_pipelines.fit_transform(housing)# neural networks
X_test = num_pipelines.fit_transform(X_test)



classifier = OneVsRestClassifier(
    svm.SVC(kernel="linear", probability=True, random_state= np.random.RandomState(0))
)
y_score = classifier.fit(housing, housing_labels).decision_function(X_test)
housing_labels = label_binarize(housing_labels, classes=[0, 1, 2,3,4,5])
y_test = label_binarize(y_test, classes=[0, 1, 2,3,4,5])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['blue', 'red', 'green','yellow','black','orange'])
for i, color in zip(range(6), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

file1 = open('myfile.txt', 'w')
file1.writelines(L)
file1.close()   

############  fasttext for the sytem call category
### scc - min - 2456 (min_count)
### scc - median - 2456 (min_count)
### scc - average - 566 (min_count)
### scc - max - 3 (min_count)

import glob, os, re
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import pickle
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

Train_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scc\\train_malware"
Train_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scc\\train_benign"
Test_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scc\\test_malware"
Test_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scc\\test_benign"


def readFile(filePath):
    with open(filePath, 'r' , encoding='utf-8') as f:
        return(  f.read().split("\n") )

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def getVocabulary(path):
    os.chdir(path)
    vocabulary=[]
    for infile in sorted(glob.glob('*.txt'),key=numericalSort):
        filePath = f"{path}/{infile}"
        vocab=readFile(filePath)
        if(len(vocab)<50000):
            vocabulary.append(vocab)
    return vocabulary
        
    

Train_Mal_Vocabulary  = getVocabulary(Train_Mal_Path)
Train_Beng_Vocabulary = getVocabulary(Train_Beng_Path)
Test_Mal_Vocabulary   = getVocabulary(Test_Mal_Path)
Test_Beng_Vocabulary =  getVocabulary(Test_Beng_Path)


Train_Combined_Vocabulary = Train_Mal_Vocabulary + Train_Beng_Vocabulary

Test_Combined_Vocabulary = Test_Mal_Vocabulary + Test_Beng_Vocabulary




# Test_Mal_Vocabulary   = getVocabulary(Test_Mal_Path)
# Test_Beng_Vocabulary  = getVocabulary(Test_Beng_Path)

val_mal_scc = 30000
val_ben_scc = 30000

def makeVocabularyWordVector(row,model):
    finalRow = np.zeros(shape=(30000,10),dtype='float16')
    for i,vocab in enumerate(row):
      xcv = model.wv[vocab]
      xcv = xcv.astype("float16")
      finalRow[i]=xcv
    return finalRow
    #return []

def getWordVector(vocabulary , model):
    # vocabulary = Train_Mal_Vocabulary
    wordVector = np.zeros(shape=( len(vocabulary),30000,10),dtype='float16')
    # model = Train_Mal_Model
    # wordVector.shape
    
    for i,row in enumerate(vocabulary):
        # print(row)
        wordVector[i]=makeVocabularyWordVector(row,model)
    return wordVector



def saveModel(vocabulary , name):
    vocabulary = Train_Combined_Vocabulary
    model = FastText(vector_size=10, window=3, min_count=3)  # instantiate
    model.build_vocab(corpus_iterable= vocabulary )
    model.train(corpus_iterable= vocabulary, total_examples=len(vocabulary), epochs=10)
    pickle.dump(model, open(name, 'wb'))
    

saveModel(Train_Combined_Vocabulary , 'D:\\ist journal\\tr.pkl')
Train_Fastext_Model = pickle.load( open('D:\\ist journal\\tr.pkl', 'rb') )


max_length_mal = 0
for j in range(0,len(Train_Mal_Vocabulary)):
    if(len(Train_Mal_Vocabulary[j])>max_length_mal):
        max_length_mal = len(Train_Mal_Vocabulary[j])

max_length_ben = 0
for j in range(0,len(Train_Beng_Vocabulary)):
    if(len(Train_Beng_Vocabulary[j])>max_length_ben):
        max_length_ben = len(Train_Beng_Vocabulary[j])
        
        
max_length_all = 0
for j in range(0,len(Train_Combined_Vocabulary)):
    max_length_all = max_length_all + len(Train_Combined_Vocabulary[j])
    
max_length_all = (max_length_all/8000)/2223  
     

Train_Mal_Word_Vector = getWordVector(Train_Mal_Vocabulary, Train_Fastext_Model)
Train_Beng_Word_Vector = getWordVector(Train_Beng_Vocabulary , Train_Fastext_Model)
Test_Mal_Word_Vector = getWordVector(Test_Mal_Vocabulary , Train_Fastext_Model)
Test_Beng_Word_Vector = getWordVector(Test_Beng_Vocabulary , Train_Fastext_Model)



Train_feature =np.concatenate((Train_Mal_Word_Vector, Train_Beng_Word_Vector), axis=0)
Test_feature =np.concatenate((Test_Mal_Word_Vector, Test_Beng_Word_Vector), axis=0)


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

import tensorflow as tf
import tensorflow as tf
import random
Bilstm_Model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Attention(),
    tf.keras.layers.Dense(1, activation ='sigmoid')
])

Bilstm_Model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


Bilstm_History = Bilstm_Model.fit(x=np.array(Train_feature),y= np.array(label),epochs=10, validation_split =0.2, verbose=1)

Bilstm_Model.summary()

malLabel = np.ones(len(Train_Mal_Word_Vector) , dtype ='int16')
bengLabel = np.zeros(len(Train_Beng_Word_Vector),  dtype ='int16' )
malfinalLabel = np.ones(len(Test_Mal_Word_Vector) , dtype ='int16')
bengfinalLabel = np.zeros(len(Test_Beng_Word_Vector),  dtype ='int16' )
    print('here we are')

label=np.concatenate((malLabel, bengLabel), axis=0)
labelfinalpredicted=np.concatenate((malfinalLabel, bengfinalLabel), axis=0)
feature =np.concatenate((Train_Mal_Word_Vector, Train_Beng_Word_Vector), axis=0)

############  fasttext for the sytem call without category
### scwc - min - 1473 (min_count)
### scwc - median - 409 (min_count)
### scwc - average - 93 (min_count)
### scwc - max - 20 (min_count)

import glob, os, re
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import pickle
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

Train_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scwc\\train_malware"
Train_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scwc\\train_benign"
Test_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scwc\\test_malware"
Test_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\scwc\\test_benign"


def readFile(filePath):
    with open(filePath, 'r' , encoding='utf-8') as f:
        return(  f.read().split("\n") )

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def getVocabulary(path):
    os.chdir(path)
    vocabulary=[]
    for infile in sorted(glob.glob('*.txt'),key=numericalSort):
        filePath = f"{path}/{infile}"
        vocab=readFile(filePath)
        if(len(vocab)<50000):
            vocabulary.append(vocab)
    return vocabulary
        
Train_Mal_Vocabulary  = getVocabulary(Train_Mal_Path)
Train_Beng_Vocabulary = getVocabulary(Train_Beng_Path)
Test_Mal_Vocabulary   = getVocabulary(Test_Mal_Path)
Test_Beng_Vocabulary =  getVocabulary(Test_Beng_Path)

Train_Combined_Vocabulary = Train_Mal_Vocabulary + Train_Beng_Vocabulary
Test_Combined_Vocabulary = Test_Mal_Vocabulary + Test_Beng_Vocabulary


max_length_all = 0
for j in range(0,len(Train_Combined_Vocabulary)):
    max_length_all = max_length_all + len(Train_Combined_Vocabulary[j])
    
max_length_all = (max_length_all/8000)/363 

max_length_mal = 0
for j in range(0,len(Test_Combined_Vocabulary)):
    if(len(Test_Combined_Vocabulary[j])>max_length_mal):
        max_length_mal = len(Test_Combined_Vocabulary[j])

def makeVocabularyWordVector(row,model):
    finalRow = np.zeros(shape=(30000,10),dtype='float16')
    for i,vocab in enumerate(row):
      xcv = model.wv[vocab]
      xcv = xcv.astype("float16")
      finalRow[i]=xcv
    return finalRow
    #return []

def getWordVector(vocabulary , model):
    # vocabulary = Train_Mal_Vocabulary
    wordVector = np.zeros(shape=( len(vocabulary),30000,10),dtype='float16')
    # model = Train_Mal_Model
    # wordVector.shape
    
    for i,row in enumerate(vocabulary):
        # print(row)
        wordVector[i]=makeVocabularyWordVector(row,model)
    return wordVector



def saveModel(vocabulary , name):
    vocabulary = Train_Combined_Vocabulary
    model = FastText(vector_size=10, window=2, min_count=20)  # instantiate
    model.build_vocab(corpus_iterable= vocabulary )
    model.train(corpus_iterable= vocabulary, total_examples=len(vocabulary), epochs=10)
    pickle.dump(model, open(name, 'wb'))
    

saveModel(Train_Combined_Vocabulary , 'D:\\ist journal\\tr.pkl')
Train_Fastext_Model = pickle.load( open('D:\\ist journal\\tr.pkl', 'rb') )

Train_Mal_Word_Vector = getWordVector(Train_Mal_Vocabulary, Train_Fastext_Model)
Train_Beng_Word_Vector = getWordVector(Train_Beng_Vocabulary , Train_Fastext_Model)
Test_Mal_Word_Vector = getWordVector(Test_Mal_Vocabulary , Train_Fastext_Model)
Test_Beng_Word_Vector = getWordVector(Test_Beng_Vocabulary , Train_Fastext_Model)



Train_feature =np.concatenate((Train_Mal_Word_Vector, Train_Beng_Word_Vector), axis=0)
Test_feature =np.concatenate((Test_Mal_Word_Vector, Test_Beng_Word_Vector), axis=0)


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

temp = list(zip(Train_feature, label))
random.shuffle(temp)
feature, label = zip(*temp)

import tensorflow as tf
import tensorflow as tf
import random
Bilstm_Model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Attention(),
    tf.keras.layers.Dense(1, activation ='sigmoid')
])

Bilstm_Model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


Bilstm_History = Bilstm_Model.fit(x=np.array(feature),y= np.array(label),epochs=5, validation_split =0.2, verbose=1)


malLabel = np.ones(len(Train_Mal_Word_Vector) , dtype ='int16')
bengLabel = np.zeros(len(Train_Beng_Word_Vector),  dtype ='int16' )
malfinalLabel = np.ones(len(Test_Mal_Word_Vector) , dtype ='int16')
bengfinalLabel = np.zeros(len(Test_Beng_Word_Vector),  dtype ='int16' )
    print('here we are')

label=np.concatenate((malLabel, bengLabel), axis=0)
labelfinalpredicted=np.concatenate((malfinalLabel, bengfinalLabel), axis=0)

predicted = Bilstm_Model.predict(Test_feature)


############  fasttext for the API call with category
### aoic - min - 8 (min_count)
### apic - median - 5 (min_count)
### apic - average - 4 (min_count)
### apic - max - 2 (min_count)

import glob, os, re
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import pickle
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

Train_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apic\\train_malware"
Train_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apic\\train_benign"
Test_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apic\\test_malware"
Test_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apic\\test_benign"


def readFile(filePath):
    with open(filePath, 'r' , encoding='utf-8') as f:
        return(  f.read().split("\n") )

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def getVocabulary(path):
    os.chdir(path)
    vocabulary=[]
    for infile in sorted(glob.glob('*.txt'),key=numericalSort):
        filePath = f"{path}/{infile}"
        vocab=readFile(filePath)
        if(len(vocab)<50000):
            vocabulary.append(vocab)
    return vocabulary
        
Train_Mal_Vocabulary  = getVocabulary(Train_Mal_Path)
Train_Beng_Vocabulary = getVocabulary(Train_Beng_Path)
Test_Mal_Vocabulary   = getVocabulary(Test_Mal_Path)
Test_Beng_Vocabulary =  getVocabulary(Test_Beng_Path)

Train_Combined_Vocabulary = Train_Mal_Vocabulary + Train_Beng_Vocabulary
Test_Combined_Vocabulary = Test_Mal_Vocabulary + Test_Beng_Vocabulary


max_length_all = 0
for j in range(0,len(Train_Combined_Vocabulary)):
    max_length_all = max_length_all + len(Train_Combined_Vocabulary[j])
    
max_length_all = (max_length_all/4366)/729


max_length_mal = 0
for j in range(0,len(Test_Combined_Vocabulary)):
    if(len(Test_Combined_Vocabulary[j])>max_length_mal):
        max_length_mal = len(Test_Combined_Vocabulary[j])

def makeVocabularyWordVector(row,model):
    finalRow = np.zeros(shape=(50000,10),dtype='float16')
    for i,vocab in enumerate(row):
      xcv = model.wv[vocab]
      xcv = xcv.astype("float16")
      finalRow[i]=xcv
    return finalRow
    #return []

def getWordVector(vocabulary , model):
    # vocabulary = Train_Mal_Vocabulary
    wordVector = np.zeros(shape=( len(vocabulary),50000,10),dtype='float16')
    # model = Train_Mal_Model
    # wordVector.shape
    
    for i,row in enumerate(vocabulary):
        # print(row)
        wordVector[i]=makeVocabularyWordVector(row,model)
    return wordVector



def saveModel(vocabulary , name):
    vocabulary = Train_Combined_Vocabulary
    model = FastText(vector_size=10, window=4, min_count=2)  # instantiate
    model.build_vocab(corpus_iterable= vocabulary )
    model.train(corpus_iterable= vocabulary, total_examples=len(vocabulary), epochs=10)
    pickle.dump(model, open(name, 'wb'))
    

saveModel(Train_Combined_Vocabulary , 'D:\\ist journal\\tr.pkl')
Train_Fastext_Model = pickle.load( open('D:\\ist journal\\tr.pkl', 'rb') )

Train_Mal_Word_Vector = getWordVector(Train_Mal_Vocabulary, Train_Fastext_Model)
Train_Beng_Word_Vector = getWordVector(Train_Beng_Vocabulary , Train_Fastext_Model)
Test_Mal_Word_Vector = getWordVector(Test_Mal_Vocabulary , Train_Fastext_Model)
Test_Beng_Word_Vector = getWordVector(Test_Beng_Vocabulary , Train_Fastext_Model)



Train_feature =np.concatenate((Train_Mal_Word_Vector, Train_Beng_Word_Vector), axis=0)
Test_feature =np.concatenate((Test_Mal_Word_Vector, Test_Beng_Word_Vector), axis=0)


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

temp = list(zip(Train_feature, label))
random.shuffle(temp)
feature, label = zip(*temp)

import tensorflow as tf
import tensorflow as tf
import random
Bilstm_Model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Attention(),
    tf.keras.layers.Dense(1, activation ='sigmoid')
])

Bilstm_Model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


Bilstm_History = Bilstm_Model.fit(x=np.array(feature),y= np.array(label),epochs=5, validation_split =0.2, verbose=1)


malLabel = np.ones(len(Train_Mal_Word_Vector) , dtype ='int16')
bengLabel = np.zeros(len(Train_Beng_Word_Vector),  dtype ='int16' )
malfinalLabel = np.ones(len(Test_Mal_Word_Vector) , dtype ='int16')
bengfinalLabel = np.zeros(len(Test_Beng_Word_Vector),  dtype ='int16' )
print('here we are')

label=np.concatenate((malLabel, bengLabel), axis=0)
labelfinalpredicted=np.concatenate((malfinalLabel, bengfinalLabel), axis=0)

predicted = Bilstm_Model.predict(Test_feature)

############  fasttext for the API call withoutcategory
### apiwc - min - 2 (min_count)
### apiwc - median - 1 (min_count)
### apiwc - average - 6 (min_count)
### apiwc - max - 2 (min_count)

import glob, os, re
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import pickle
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

Train_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apiwc\\train_malware"
Train_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apiwc\\train_benign"
Test_Mal_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apiwc\\test_malware"
Test_Beng_Path = "C:\\Users\\admin\\Documents\\dataset_ist_journal_faststext\\apiwc\\test_benign"


def readFile(filePath):
    with open(filePath, 'r' , encoding='utf-8') as f:
        return(  f.read().split("\n") )

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def getVocabulary(path):
    os.chdir(path)
    vocabulary=[]
    for infile in sorted(glob.glob('*.txt'),key=numericalSort):
        filePath = f"{path}/{infile}"
        vocab=readFile(filePath)
        if(len(vocab)<50000):
            vocabulary.append(vocab)
    return vocabulary
        
Train_Mal_Vocabulary  = getVocabulary(Train_Mal_Path)
Train_Beng_Vocabulary = getVocabulary(Train_Beng_Path)
Test_Mal_Vocabulary   = getVocabulary(Test_Mal_Path)
Test_Beng_Vocabulary =  getVocabulary(Test_Beng_Path)

Train_Combined_Vocabulary = Train_Mal_Vocabulary + Train_Beng_Vocabulary
Test_Combined_Vocabulary = Test_Mal_Vocabulary + Test_Beng_Vocabulary


max_length_all = 0
for j in range(0,len(Train_Combined_Vocabulary)):
    max_length_all = max_length_all + len(Train_Combined_Vocabulary[j])
    
max_length_all = (max_length_all/4366)/6919


max_length_mal = 0
for j in range(0,len(Test_Combined_Vocabulary)):
    if(len(Test_Combined_Vocabulary[j])>max_length_mal):
        max_length_mal = len(Test_Combined_Vocabulary[j])

def makeVocabularyWordVector(row,model):
    finalRow = np.zeros(shape=(50000,10),dtype='float16')
    for i,vocab in enumerate(row):
      xcv = model.wv[vocab]
      xcv = xcv.astype("float16")
      finalRow[i]=xcv
    return finalRow
    #return []

def getWordVector(vocabulary , model):
    # vocabulary = Train_Mal_Vocabulary
    wordVector = np.zeros(shape=( len(vocabulary),50000,10),dtype='float16')
    # model = Train_Mal_Model
    # wordVector.shape
    
    for i,row in enumerate(vocabulary):
        # print(row)
        wordVector[i]=makeVocabularyWordVector(row,model)
    return wordVector



def saveModel(vocabulary , name):
    vocabulary = Train_Combined_Vocabulary
    model = FastText(vector_size=10, window=4, min_count=2)  # instantiate
    model.build_vocab(corpus_iterable= vocabulary )
    model.train(corpus_iterable= vocabulary, total_examples=len(vocabulary), epochs=10)
    pickle.dump(model, open(name, 'wb'))
    

saveModel(Train_Combined_Vocabulary , 'D:\\ist journal\\tr.pkl')
Train_Fastext_Model = pickle.load( open('D:\\ist journal\\tr.pkl', 'rb') )

Train_Mal_Word_Vector = getWordVector(Train_Mal_Vocabulary, Train_Fastext_Model)
Train_Beng_Word_Vector = getWordVector(Train_Beng_Vocabulary , Train_Fastext_Model)
Test_Mal_Word_Vector = getWordVector(Test_Mal_Vocabulary , Train_Fastext_Model)
Test_Beng_Word_Vector = getWordVector(Test_Beng_Vocabulary , Train_Fastext_Model)



Train_feature =np.concatenate((Train_Mal_Word_Vector, Train_Beng_Word_Vector), axis=0)
Test_feature =np.concatenate((Test_Mal_Word_Vector, Test_Beng_Word_Vector), axis=0)


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

temp = list(zip(Train_feature, label))
random.shuffle(temp)
feature, label = zip(*temp)

import tensorflow as tf
import tensorflow as tf
import random
Bilstm_Model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Attention(),
    tf.keras.layers.Dense(1, activation ='sigmoid')
])

Bilstm_Model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

Bilstm_Model.summary()

Bilstm_History = Bilstm_Model.fit(x=np.array(feature),y= np.array(label),epochs=5, validation_split =0.2, verbose=1)


malLabel = np.ones(len(Train_Mal_Word_Vector) , dtype ='int16')
bengLabel = np.zeros(len(Train_Beng_Word_Vector),  dtype ='int16' )
malfinalLabel = np.ones(len(Test_Mal_Word_Vector) , dtype ='int16')
bengfinalLabel = np.zeros(len(Test_Beng_Word_Vector),  dtype ='int16' )
print('here we are')

label=np.concatenate((malLabel, bengLabel), axis=0)
labelfinalpredicted=np.concatenate((malfinalLabel, bengfinalLabel), axis=0)

predicted = Bilstm_Model.predict(Test_feature)

    
########## training xgboost
import xgboost as xgb 
import time  
from sklearn.metrics import mean_squared_error
classifier_xgb = xgb.XGBClassifier(n_estimators = 100,learning_rate=0.1,max_depth=3)
from dask import dataframe as dd
start = time.time()


df1 = pd.read_csv(r"C:\Users\admin\Documents\Scie_output\scwcmax2gram+final.csv",low_memory=False)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tester = df1['label']

df1 = df1.drop("label",axis=1)
df1['label']=templabletransfrecsv["label"]
df1['label_updated']=templabletransfrecsv["label_updated"]

for train_index, test_index in split.split(df1,df1["label"]):
    strat_train_set = df1.iloc[train_index]
    strat_test_set = df1.iloc[test_index]
# Looking for correltaions
housing = strat_train_set.copy()
housing = housing.drop("label_updated", axis=1) 
housing = housing.drop("label", axis=1) 
housing_labels = strat_train_set["label_updated"].copy()
counter = Counter(housing_labels)
print(counter)
housing_test = strat_test_set.copy()
housing_test = housing_test.drop("label_updated", axis=1) 
housing_test = housing_test.drop("label", axis=1) 
y_test = strat_test_set["label_updated"].copy()
counter = Counter(y_test)
print(counter)

classifier_xgb.fit(housing,housing_labels)
y_pred = classifier_xgb.predict(housing_test)
metrics.confusion_matrix(y_test,y_pred)
mean_squared_error(y_test,y_pred)

####### 
path1 = "C:\\Users\\admin\\Documents\\scie\\"
path2 = "C:\\Users\\admin\\Documents\\sccate\\"
lis1 =[]
lis2=[]
def openFile(fileName):
    fileHere = os.path.join(path2,fileName)
    with open(fileHere, 'r' , encoding='utf-8') as f:
        lis2.append( f.read().splitlines())

files = os.listdir(path2)
for file in files:
    openFile(file)
    
lis1_final=[]
for k in range(0,len(lis1)):
    for z in range(0,len(lis1[k])):
        
        lis1_final.append(lis1[k][z])
    
lis2_final=[]
for k in range(0,len(lis2)):
    for z in range(0,len(lis2[k])):
        
        lis2_final.append(lis2[k][z])

test_keys = ["Rash","Rash", "Kil", "Varsha"]
test_values = [1,1,4,5]
  
# Printing original keys-value lists
print ("Original key list is : " + str(test_keys))
print ("Original value list is : " + str(test_values))
  
# using naive method
# to convert lists to dictionary
res = {}
cpic = 0
for key in lis1_final:
    print(cpic)
    cpic = cpic+1
    
    for value in lis2_final:
        res[key] = value
        lis2_final.remove(value)
        break  
  
# Printing resultant dictionary 
print ("Resultant dictionary is : " +  str(res))

index_ava=[]
for k in range(0,len(lis2_final)):
    if(lis2_final[k]=="virtual"):
        index_ava.append(k)

lis2[1]

pikachu =[]
for k in index_ava:
    pikachu.append(lis1_final[k])


pikachu_set = set(pikachu)

lis1_final[72072192]


unique_sys = set(lis2_final)



###### finding number of samples in each class of microsoft malware challenge
read_label = pd.read_csv(r"D:\malware-classification\testLabels.csv")
Counter(read_label['Class'])

##############  BIN-SIZE LOCATION
len(lis[0])
fibin=[]
for i in range(0,len(lis)):
    fibin.append(len(lis[i]))
    
fibin

maxHere = max(fibin)
minHere = min(fibin)
avgHere = statistics.mean(fibin)
medianhere = statistics.median(fibin)

########### MIN_COOUNT EVALUATE
fl1p= pd.read_csv(r"C:\Users\admin\Documents\Scie_output\apic_min_1gram.csv")
fl1p= pd.read_csv(r"C:\Users\admin\Documents\Scie_output\apiwc_min_1gram.csv")
fl1p= pd.read_csv(r"C:\Users\admin\Documents\Scie_output\scc_min_1gram.csv")
fl1p= pd.read_csv(r"C:\Users\admin\Documents\Scie_output\syswc_min_1gram.csv")
                                                                                     
fry=[]        
for p in range(0,len(fl1p)):
    for j in range(0,len(fl1p.iloc[p])-1):
        fry.append(fl1p.iloc[p][j])
        

list_without_zeros = [x for x in fry if x != 0]

max(list_without_zeros)
min(list_without_zeros)
statistics.mean(list_without_zeros)
statistics.median(list_without_zeros)


#### applying cnn1



from tensorflow import keras
df1 = pd.read_csv("C:\\Users\\admin\\Documents\\Scie_output\\sccmin3gram+final.csv",low_memory=False)
for z in range(0,8):
    sp = "uop"+str(z)
    df1[sp]=0
df1 = df1.drop("label",axis=1)
df1['label']=templabletransfrecsv["label"]
df1['label_updated']=templabletransfrecsv["label_updated"]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df1,df1["label"]):
    strat_train_set = df1.iloc[train_index]
    strat_test_set = df1.iloc[test_index]
# Looking for correltaions
housing = strat_train_set.copy()
housing = housing.drop("label_updated", axis=1) 
housing = housing.drop("label", axis=1) 
housing_labels = strat_train_set["label_updated"].copy()
counter = Counter(housing_labels)
print(counter)
housing_test = strat_test_set.copy()
housing_test = housing_test.drop("label_updated", axis=1) 
housing_test = housing_test.drop("label", axis=1) 
y_test = strat_test_set["label_updated"].copy()
housing = housing.to_numpy()
# avast = housing.reshape((4779,347,347))
# avast = housing.reshape((6896,32,32))
# avast = housing.reshape((6896,10,10))
# avast = housing.reshape((6896,32,32))
# avast = housing.reshape((6896,40,40))
avast = housing.reshape((4779,83,83))
avast = avast[...,np.newaxis]
X_test = housing_test.to_numpy()
# avast_test = X_test.reshape((1195,347,347))
# avast_test = X_test.reshape((1725,32,32))
# avast_test = X_test.reshape((1725,10,10))
# avast_test = X_test.reshape((1725,40,40))
avast_test = X_test.reshape((1195,83,83))
avast_test=avast_test[...,np.newaxis]
ulabel_codes = pd.Categorical(housing_labels).codes
ulabel_codes = pd.Series(ulabel_codes)
ulabeltest_codes = pd.Categorical(y_test).codes
ulabeltest_codes = pd.Series(ulabeltest_codes)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 6, 
                    kernel_size = 5, 
                    strides = 1, 
                    activation = 'relu', 
                    input_shape = (40,40,1)))
model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
model.add(keras.layers.Conv2D(filters = 16, 
                 kernel_size = 2, 
                 strides = 1, 
                 activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units = 120, activation = 'relu'))
model.add(keras.layers.Dense(units = 84, activation = 'relu'))
model.add(keras.layers.Dense(units = 2, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(avast,ulabel_codes, epochs=10, validation_split=0.3)
pred = model.predict(avast_test)
fpl=[]
for j in range(0,len(pred)):
    fpl.append(np.argmax(pred[j]))
    
metrics.confusion_matrix(y_test,y_pred)


###### microsoft dataset work
path = "D:\\malware-classification\\train\\train_asm_api\\"    
pathtest = "D:\\malware-classification\\test\\test_asm_api\\"
import glob
import os

os.chdir(path)
my_files = glob.glob('*.txt')
os.chdir(pathtest)
my_files_test = glob.glob('*.txt')
lbcount = 0  
## dict microsoft
dictcasv = pd.read_csv("D:\\malware-classification\\trainLabels.csv")
dictcasv_test = pd.read_csv("D:\\malware-classification\\testLabels.csv")
my_files1=[]
for k in range(0,len(my_files)):
    print(k)
    name = my_files[k].replace('api', '')
    name1 = name.replace('.txt','')
    refuindex=list(dictcasv.iloc[:,0]).index(name1)
    labelupdate = dictcasv.iloc[refuindex,1]
   
    fileHeres = os.path.join(path,my_files[k])
    # print(fileHeres)
    lis=[]
    try:
        with open(fileHeres, 'r' , encoding='utf-8') as f:
            lis.append( f.read().splitlines())
        listowrite=[]
        for s in range(0,len(lis[0])):
            listowrite.append(lis[0][s].replace(':dword', '').lower())
        if(labelupdate==1):
            nametowrite="ramnit"
        elif(labelupdate==2):
            nametowrite="lollipop"
        elif(labelupdate==3):
            nametowrite="kelihosver3"
        elif(labelupdate==4):
            nametowrite="vundo"
        elif(labelupdate==5):
            nametowrite="simda"
        elif(labelupdate==6):
            nametowrite="tracur"
        elif(labelupdate==7):
            nametowrite="kelihosver1"
        elif(labelupdate==8):
            nametowrite="obfuscaator"
        elif(labelupdate==9):
            nametowrite="gatak"
        lbcount = lbcount +1  
        namelb= nametowrite + str(lbcount)+".txt"
        with open(namelb,'w') as fp:
            fp.write('\n'.join(listowrite))    
        f.close()    

    
      
    except IOError:
        print(fileHeres)


for k in range(0,len(my_files_test)):
    print(k)
    # name = my_files_test[k].replace('api', '')
    name1 = my_files_test[k].replace('.txt','')
    refuindex=list(dictcasv_test.iloc[:,0]).index(name1)
    labelupdate = dictcasv_test.iloc[refuindex,1]
   
    fileHeres = os.path.join(pathtest,my_files_test[k])
    # print(fileHeres)
    lis=[]
    try:
        with open(fileHeres, 'r' , encoding='utf-8') as f:
            lis.append( f.read().splitlines())
        listowrite=[]
        for s in range(0,len(lis[0])):
            listowrite.append(lis[0][s].replace(':dword', '').lower())
        if(labelupdate==1):
            nametowrite="ramnit"
        elif(labelupdate==2):
            nametowrite="lollipop"
        elif(labelupdate==3):
            nametowrite="kelihosver3"
        elif(labelupdate==4):
            nametowrite="vundo"
        elif(labelupdate==5):
            nametowrite="simda"
        elif(labelupdate==6):
            nametowrite="tracur"
        elif(labelupdate==7):
            nametowrite="kelihosver1"
        elif(labelupdate==8):
            nametowrite="obfuscaator"
        elif(labelupdate==9):
            nametowrite="gatak"
        lbcount = lbcount +1  
        namelb= nametowrite + str(lbcount)+".txt"
        with open(namelb,'w') as fp:
            fp.write('\n'.join(listowrite))    
        f.close()    

    
      
    except IOError:
        print(fileHeres)
        
    
# calls avilable in microsoft
import glob
import os
path = "D:\\malware-classification\\train\\train_categorical_api\\"
os.chdir(path)
my_files = glob.glob('*.txt')
pathtry = "D:\\malware-classification\\test\\test_categorical_api\\"
os.chdir(pathtry)
my_files_try = glob.glob('*.txt')
lis=[]
for k in range(0,len(my_files)):
    print(k)
    fileHeres = os.path.join(path,my_files[k])
    # print(fileHeres)
   
  
    with open(fileHeres, 'r' , encoding='utf-8') as f:
        lis.append( f.read().splitlines())
listowrite=[]
for s in range(0,len(lis)):
    for z in range(0,len(lis[s])):
        listowrite.append(lis[s][z])
        
listowrite=set(listowrite)   
listowrite = list(listowrite)


lis=[]
for k in range(0,len(my_files_try)):
    print(k)
    fileHeres = os.path.join(pathtry,my_files_try[k])
    # print(fileHeres)
   
  
    with open(fileHeres, 'r' , encoding='utf-8') as f:
        lis.append( f.read().splitlines())
for s in range(0,len(lis)):
    for z in range(0,len(lis[s])):
        listowrite.append(lis[s][z])
        
listowrite=set(listowrite)    
listowrite = list(listowrite) 
with open("listofcategoricalapi.txt",'w') as fp:
    fp.write('\n'.join(listowrite))    

###### converting to categorical microsoft
dictimicro = pd.read_csv("C:\\Users\\admin\\Documents\\Scie_output\\categorical_calls (Autosaved).csv")
dictmicro = list(zip(dictimicro['Calls'],dictimicro['Category']))
dictmicro = dict(dictmicro)
dictmicro['unk_4ce000']

import glob
import os
path = "D:\\malware-classification\\train\\train_categorical_api"
pathtest = "D:\\malware-classification\\test\\test_categorical_api"
os.chdir(path)
os.chdir(pathtest)
my_files = glob.glob('*.txt')
pathtry = "D:\\malware-classification\\test\\test_categorical_api\\"
os.chdir(path)
my_files = glob.glob('*.txt')
lis=[]
for k in range(0,len(my_files)):
    print(k)
    fileHeres = os.path.join(pathtest,my_files[k])
    # print(fileHeres)
   
  
    with open(fileHeres, 'r' , encoding='utf-8') as f:
        lis.append( f.read().splitlines())

def replace(list, dictionary):
    return [dictmicro.get(item, item) for item in list]

lis2 = []
for x in range(0,len(lis)):
    rtp= lis[x]
    lis2.append(replace(rtp,dictmicro))

for x in range(0,len(lis2)): 
    nameoflist = my_files[x]
    with open(nameoflist,'w') as fp:
        for items in lis2[x]:
            fp.write('%s\n' %items)   
    fp.close()  
    
    
#### applying cnn1 microsoft

from tensorflow import keras
df1 = pd.read_csv("D:\\malware-classification\\apicmax4gram+microsoft+final+train.csv",low_memory=False)
df2 = pd.read_csv("D:\\malware-classification\\apicmax4gram+microsoft+final+test.csv",low_memory=False)
df2 = pd.read_csv("C:\\Users\\admin\\Documents\\Scie_output\\malgansc_virustotal.csv",low_memory=False)
for z in range(0,94):
    sp = "uop"+str(z)
    df1[sp]=0
    df2[sp]=0

housing = df1.copy()
housing = housing.drop("label_updated", axis=1) 
housing = housing.drop("label", axis=1) 
housing_labels = df1["label_updated"].copy()
counter = Counter(housing_labels)
print(counter)


housing_test = df2.copy()
housing_test = housing_test.drop("label_updated", axis=1) 
housing_test = housing_test.drop("label", axis=1) 
y_test = df2["label_updated"].copy()
counter = Counter(y_test)
print(counter)

housing = housing.to_numpy()
# avast = housing.reshape((4779,347,347))
# avast = housing.reshape((6896,32,32))
# avast = housing.reshape((6896,10,10))
# avast = housing.reshape((6896,32,32))
# avast = housing.reshape((6896,40,40))
avast = housing.reshape((8621,32,32))
#avast = housing.reshape((10404,63,63))
avast = avast[...,np.newaxis]
X_test = housing_test.to_numpy()
# avast_test = X_test.reshape((1195,347,347))
# avast_test = X_test.reshape((1725,32,32))
# avast_test = X_test.reshape((1725,10,10))
# avast_test = X_test.reshape((1725,40,40))
avast_test = X_test.reshape((2000,32,32))
avast_test=avast_test[...,np.newaxis]
ulabel_codes = pd.Categorical(housing_labels).codes
ulabel_codes = pd.Series(ulabel_codes)
ulabeltest_codes = pd.Categorical(y_test).codes
ulabeltest_codes = pd.Series(ulabeltest_codes)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 6, 
                    kernel_size = 5, 
                    strides = 1, 
                    activation = 'relu', 
                    input_shape = (32,32,1)))
model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
model.add(keras.layers.Conv2D(filters = 16, 
                 kernel_size = 2, 
                 strides = 1, 
                 activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units = 120, activation = 'relu'))
model.add(keras.layers.Dense(units = 84, activation = 'relu'))
model.add(keras.layers.Dense(units = 2, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(avast,ulabel_codes, epochs=10, validation_split=0.3)
pred = model.predict(avast_test)
fpl=[]
for j in range(0,len(pred)):
    fpl.append(np.argmax(pred[j]))
    
metrics.confusion_matrix(ulabeltest_codes,fpl)
    