import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

def make_prediction(email):
    tokenized_email = tokenizer.transform([email])
    prediction = model.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return prediction