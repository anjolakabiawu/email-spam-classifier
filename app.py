from flask import Flask, render_template, request
import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get("email-content")
    tokenized_email = tokenizer.transform([email])
    prediction = model.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True) # putting the debug=True is for making updates go through on the site without having to restart the app manually