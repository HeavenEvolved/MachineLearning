from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#import warnings
#warnings.filterwarnings('ignore')

scaler = StandardScaler()

def predictor(model, data_df):

    X_trained = pd.read_csv('./X_trained.csv', index_col=None)

    X_trained_scaled = scaler.fit_transform(X_trained)
    data_df_scaled = scaler.transform(data_df)

    #data_np = data_df.to_numpy(dtype="float32")

    prediction = model.predict(data_df_scaled)

    return prediction

model_in = load('model.joblib')

app = Flask(__name__, static_folder='./static/')
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/", methods=['GET', 'POST'])
def main():
    request_type = request.method

    if request_type == "GET":

        return render_template('index.html')

    elif request_type == "POST":

        pclass = request.form['pclass']
        sex = request.form['sex']
        age = request.form['age']
        fare = request.form['fare']
        embarked = request.form['embarked']
        relatives = request.form['relatives']
        alone = 1 if relatives == 0 else 0

        data = [pclass, sex, age, fare, embarked, relatives, alone]

        data_df = pd.DataFrame([data], columns=['Pclass', 'Sex', 'Age', 'Fare','Embarked', 'Relatives', 'Alone'])

        pred = predictor(model_in, data_df)

        prediction_string = "Probability of Survival: " + str(pred[0])

        return render_template('index.html', data=[data_df.to_html(classes='data', header="false")], pred_str=prediction_string)

"""{% for table in pred_str %}
{{ table|safe }}
{% endfor %}"""