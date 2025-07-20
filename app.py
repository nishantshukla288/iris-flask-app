from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("iris.data", names=["sepal length", "sepal width", "petal length", "petal width", "class"])
features = df.iloc[:, 0:4]
target = df.iloc[:, -1]

label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

# Train model
clf = DecisionTreeClassifier()
clf.fit(features, target)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])
            pred = clf.predict([[sl, sw, pl, pw]])
            prediction = label_encoder.classes_[pred[0]]
        except:
            prediction = "Invalid input!"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Use PORT from environment for deployment
    app.run(host="0.0.0.0", port=port)

