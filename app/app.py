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


@app.route('/index')
def get_app():
    """
    Renders the main application page.

    Returns:
        str: The HTML content of the rendered page.
    """
    css_url = url_for('static', filename='style.css')
    return render_template('./index.html')


@app.route('/model-1')
def model_1():
    """
    Renders the Model 1 page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('model_1.html')


@app.route('/contact')
def contact_page():
    """
    Renders the Infos page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('contact.html')


@app.route('/model-2')
def info_model_2():
    """
    Renders the Model 2 page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('model_2.html')


@app.route('/info-mod-1-page-1')
def info_model_1_page_1():
    """
    Renders the Infos about the content of the Model 1 page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('info_model_1_page_1.html')


@app.route('/info-mod-1-page-2')
def info_model_1_page_2():
    """
    Renders the Infos about the content of the Model 1 page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('info_model_1_page_2.html')


@app.route('/info-model-2')
def model_2():
    """
    Renders the Infos about the content of the Model 2 page.

    Returns:
        str: The HTML content of the rendered page.
    """
    return render_template('info_model_2.html')


@app.route('/prediction', methods=['POST', 'GET'])
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

    if request.method == 'POST':
        input_data = {}
        for key in request.form:
            input_data[key] = request.form[key]
        data = m1u.get_input_values(input_data)
        print(input_data)
        text_prediction = m1u.predict_cervical_cancer_risk(data)
        return render_template('prediction.html', prediction=text_prediction)
    else:
        return render_template('prediction.html')


if __name__ == '__main__':
    app.run(port=3000, debug=True)
