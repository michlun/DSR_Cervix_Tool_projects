from flask import Flask, render_template, url_for, send_from_directory, request
import model_1_utils as m1u
import os

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def get_app():
    css_url = url_for('static', filename='style.css')
    return render_template('./index.html')


@app.route('/index.html')
def back_home():
    return render_template('index.html')


@app.route('/model_1.html')
def model_1():
    return render_template('model_1.html')


@app.route('/model_2.html')
def model_2():
    return render_template('model_2.html')


@app.route('/prediction.html')
def prediction_page():
    return render_template('prediction.html')


@app.route('/predictdata', methods=['POST', 'GET'])
def predict_data():
    if request.method == 'GET':
        return render_template('model_1.html')
    else:
        input_data = {}
        for key in request.form:
            input_data[key] = request.form[key]
        data = m1u.get_input_values(input_data)
        prediction = m1u.predict_cervical_cancer_risk(data)
        return prediction


if __name__ == '__main__':
    app.run(port=3000, debug=True)
