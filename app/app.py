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

Author: [Francesco & Michele]
"""

from risk_prediction import model_1_utils as m1u
from flask import Flask, render_template, url_for, request

from flask import Flask, render_template, url_for, send_from_directory, request
import base64
from PIL import Image
import io
from cell_detection.model_2_utils import predict_image_class, class_recall, class2_recall
from tensorflow import keras
from keras.models import load_model

# Load prediction models for images
model_whole = 'cell_detection/conv1_192x256_lr001_1dense256.h5'
model_whole = load_model(model_whole)
model_cell = 'cell_detection/cell_conv1_aug_80x80_1dense128.h5'
model_cell = load_model(model_cell, custom_objects={"class2_recall": class2_recall})


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
def info_model_2():
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


@app.route('/model-2', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image file
        image = request.files['image']

        # convert image to base64 string
        image_string = base64.b64encode(image.read()).decode('utf-8')

        # convert to PIL format and compute scale factor
        image = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image))
        scale_fac = image.size[0] / 1024.
        # print('Scale factor:', scale_fac)

        # Setting image type for prediction and rendering:
        image_type = 'Whole slide'

        # inference
        prediction, confidence, heatmap = predict_image_class(model=model_whole,
                                                              image=image,
                                                              image_type=image_type,
                                                              gradcam_map=True)

        # Convert confidence to percentual string and a parameter for rendering
        confidence_string = str(round(confidence * 100, 2))
        whole_slide = True

        # Convert heatmap to pillow and then base64 format
        heatmap_pil = Image.fromarray(heatmap)
        buffered = io.BytesIO()
        heatmap_pil.save(buffered, format="JPEG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Render the results in a page along with the image
        return render_template('results.html',
                               prediction=prediction,
                               confidence=confidence_string,
                               image=image_string,
                               image_type=image_type,
                               img_scale=scale_fac,
                               heatmap=heatmap_b64,
                               whole_slide=whole_slide)

    return render_template('upload.html')


@app.route('/annotate', methods=['POST'])
def annotate_file():
    # Retrieve the annotated region from the form submission
    xstart = int(request.form['xstart'])
    xend = int(request.form['xend'])
    ystart = int(request.form['ystart'])
    yend = int(request.form['yend'])
    image_base64 = request.form['image']
    # print('Xstart', xstart, 'Xend', xend, 'Ystart', ystart, 'Yend', yend)

    # crop the image based on the annotated region
    image = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image))
    width, height = image.size
    # print('Uncropped image size:', image.size)
    # new_width = xend - xstart
    # new_height = yend - ystart
    image = image.crop((xstart, ystart, xend, yend))
    # image = image.resize((new_width, new_height))
    # print('Cropped image size:', image.size)

    # convert pillow image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Setting image type for prediction and rendering
    image_type = 'Single cell'

    # Inference
    prediction, confidence, heatmap = predict_image_class(model=model_cell,
                                                          image=image,
                                                          image_type=image_type,
                                                          gradcam_map=True)

    # Convert confidence to percentual string
    confidence_string = str(round(confidence * 100, 2))

    # Convert heatmap to pillow and then base64 format
    heatmap_pil = Image.fromarray(heatmap)
    buffered = io.BytesIO()
    heatmap_pil.save(buffered, format="JPEG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Render the results in a page along with the single cell image
    return render_template('results.html',
                           image=image_base64,
                           prediction=prediction,
                           confidence=confidence_string,
                           image_type=image_type,
                           img_scale=1,
                           heatmap=heatmap_b64)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
