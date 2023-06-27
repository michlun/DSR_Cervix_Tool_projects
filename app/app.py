"""
This script implements a Flask web application for a cervical cancer prediction model.

The application uses Flask framework to define various routes that serve HTML templates and handle form submissions.
The templates are rendered using the Jinja templating engine, and the predictions are made using the functions
defined in the 'model_1_utils' module.

Routes:
- '/' or '/index.html': Renders the home page template ('index.html').
- '/model_1.html': Renders the template for Model 1.
- '/model_2.html': Renders the template for Model 2.
- '/prediction.html': Renders the template for the prediction page.
- '/predictdata': Handles the form submission for prediction. Accepts both GET and POST methods.
                   If the method is GET, it renders the Model 1 template. If the method is POST,
                   it retrieves the input data from the form, calls the 'get_input_values' function
                   to extract the input values, and then calls the 'predict_cervical_cancer_risk' function
                   to make a prediction using the extracted data. The prediction result is returned as a response.

Note: The application assumes the existence of HTML templates ('index.html', 'model_1.html', 'model_2.html',
      'prediction.html') and a module named 'model_1_utils' that contains the necessary functions.

Author: [Francesco]
"""

from flask import Flask, render_template, url_for, send_from_directory, request
import model_1_utils as m1u
import os

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/index.html')
def get_app():
    """
    Renders the main application page.

    Returns:
        str: The HTML content of the rendered page.
    """
    css_url = url_for('static', filename='style.css')
    return render_template('./index.html')


@app.route('/model_1.html')
def model_1():
    """
    Renders the Model 1 page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('model_1.html')


@app.route('/model_2.html')
def model_2():
    """
    Renders the Model 2 page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('model_2.html')


@app.route('/prediction.html')
def prediction_page():
    """
    Renders the prediction page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('prediction.html')


@app.route('/predictdata', methods=['POST', 'GET'])
def predict_data():
    """
    Handles the prediction data.

    If the request method is GET, renders the Model 1 page.

    If the request method is POST, extracts input data from the request form,
    processes the data using `m1u.get_input_values` function, and predicts the
    cervical cancer risk using `m1u.predict_cervical_cancer_risk` function.

    Returns:
        str: The prediction result or the HTML content of the Model 1 page.
    """
    if request.method == 'GET':
        return render_template('/model_1.html')
    else:
        input_data = {}
        for key in request.form:
            input_data[key] = request.form[key]
        data = m1u.get_input_values(input_data)
        prediction = m1u.predict_cervical_cancer_risk(data)
        text_prediction = f"The prediction is {prediction}"
        return render_template('prediction.html', prediction=text_prediction)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
