from flask import Flask, render_template, request
import pickle

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    email_text = ""
    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()  # strip whitespace
        if email_text:  # only predict if there is text
            X = vectorizer.transform([email_text])
            proba = model.predict_proba(X)[0][1]
            prediction = round(proba, 3)
        else:
            prediction = None  # or show a message like "Please enter text"
    return render_template("index.html", prediction=prediction, email_text=email_text)


if __name__ == "__main__":
    app.run(debug = True)





        
    




