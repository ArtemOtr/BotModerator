from data_preproc import data_bow, y, data2_bow, y2
from sklearn.linear_model import LogisticRegression
import joblib




model = LogisticRegression(C=100, max_iter=500, penalty='l2') #узнали при помощи grid_search в ноутбуке
model.fit(data_bow, y)
joblib.dump(model, 'logistic_regression_toxic.pkl')

model = LogisticRegression(C=100, max_iter=500, penalty='l2') #узнали при помощи grid_search в ноутбуке
model.fit(data2_bow, y2)
joblib.dump(model, 'logistic_regression_spam.pkl')


