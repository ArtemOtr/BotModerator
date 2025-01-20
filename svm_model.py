from data_preproc import data_bow, y, data_tfidf, y2
from sklearn.svm import LinearSVC
import joblib

model = LinearSVC(C = 1, max_iter=1000, penalty='l2')
model.fit(data_bow, y)
joblib.dump(model, 'svm_toxic.pkl')


model = LinearSVC(C = 1, max_iter=1000, penalty='l2')
model.fit(data_tfidf, y2)
joblib.dump(model, 'svm_spam.pkl')