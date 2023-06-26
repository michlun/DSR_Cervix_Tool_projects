from flask import Flask, render_template, url_for, send_from_directory, request
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
def prediction():
    return render_template('prediction.html')


@app.route('/save', methods=['POST'])
def save_data():
    input_data = {}
    for key in request.form:
        input_data[key] = request.form[key]
    return input_data


if __name__ == '__main__':
    app.run(port=3000, debug=True)
