import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix


df_train = pd.read_json("hf://datasets/AlexSham/Toxic_Russian_Comments/train.jsonl", lines=True)
df_test = pd.read_json("hf://datasets/AlexSham/Toxic_Russian_Comments/test.jsonl", lines=True)

X = df_train['text']
y = df_train['label']
X_test = df_test['text']
y_test = df_test['label']

X = [x.lower() for x in X]
X_test = [x.lower() for x in X_test]

data = pd.concat([pd.Series(X), pd.Series(X_test)], axis = 0)
y  = pd.concat([y,y_test], axis = 0 )

'''
maxLen = 0
for x in data:
  if len(x.split()) > maxLen:
    maxLen = len(x.split())
print(maxLen) 
'''

count_vectorizer = CountVectorizer()
data_bow = count_vectorizer.fit_transform(data)

# tf-idf на моделях показал результаты хуже чем CountVectorizer

data_bow = csr_matrix(data_bow)






df2 = pd.read_csv("hf://datasets/DmitryKRX/anti_spam_ru/df.csv")
df2 = df2.dropna(subset=['text'])
X2 = df2['text']
X2 = X2.fillna('')
y2 = df2['is_spam']

count_vectorizer2 = CountVectorizer()
tf_idf_vectorizer = TfidfVectorizer()

data2_bow = count_vectorizer2.fit_transform(X2)
data_tfidf = tf_idf_vectorizer.fit_transform(X2)
data2_bow = csr_matrix(data2_bow)
data_tfidf = csr_matrix(data_tfidf)


