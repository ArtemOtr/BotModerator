from data_preproc import data_bow, y, data2_bow, y2

from sklearn.naive_bayes import MultinomialNB
import joblib

model = MultinomialNB(alpha = 0.7)
model.fit(data_bow, y)
joblib.dump(model, 'naive_bayes_toxic.pkl')

model = MultinomialNB(alpha = 0.7)
model.fit(data2_bow, y2)
joblib.dump(model, 'naive_bayes_spam.pkl')