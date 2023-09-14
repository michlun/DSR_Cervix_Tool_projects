from risk_prediction import model_1_utils as m1u
from flask import Flask, render_template, url_for, send_from_directory, request
import base64
from PIL import Image
import io
from cell_detection.model_2_utils import predict_image_class, class_recall, class2_recall
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19


# Cell detection model files 
model_whole_name = 'cell_detection/vgg19_globavgpool_drop_1dense256_finetune_3.h5'
model_cell_name = 'cell_detection/vgg19_128x128_globalavgpool_1dense128_finetune_3.h5'

model_whole = None 
model_cell = None

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
    """
    Handles the upload of images (GET) and the prediction of whole slide images
    """

    global model_whole_name
    global model_whole

    if request.method == 'POST':
        # Get the uploaded image file
        image = request.files['image']

        # Convert image to base64 string
        image_string = base64.b64encode(image.read()).decode('utf-8')

        # Convert to PIL format and compute scale factor
        image = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image))
        scale_fac = image.size[0] / min(1024., image.size[0])

        # Setting image type for prediction and rendering and load the model if required
        image_type = 'Whole slide'
        if model_whole == None:
            model_whole = load_model(model_whole_name)
        
        # Preprocessing the image for prediciton
        image_resized = image.resize((256,192), resample=Image.Resampling.BILINEAR)
        img_array = img_to_array(image_resized)
        img_array = preprocess_input_vgg19(img_array)

        # Inference
        prediction, confidence, heatmap = predict_image_class(model=model_whole,
                                                              image=img_array,
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
    """
    Handles the selection of an area in the whole slide images
    and the prediction of the single cell
    """

    global model_cell_name
    global model_cell

    # Retrieve the annotated region from the form submission
    xstart = int(request.form['xstart'])
    xend = int(request.form['xend'])
    ystart = int(request.form['ystart'])
    yend = int(request.form['yend'])
    image_base64 = request.form['image']

    # Crop the image based on the annotated region
    image = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image))
    width, height = image.size
    image = image.crop((xstart, ystart, xend, yend))

    # Compute scale factor
    min_width = max(width, 128.)
    scale_fac = width/ min(1536., min_width)
    
    # Convert pillow image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Setting image type for prediction and rendering and load the model if required
    image_type = 'Single cell'
    if model_cell == None:
        model_cell = load_model(model_cell_name, custom_objects={"class2_recall": class2_recall})
    
    # Preprocessing the image for prediciton
    image_resized = image.resize((128,128), resample=Image.Resampling.BILINEAR)
    img_array = img_to_array(image_resized)
    img_array = preprocess_input_vgg19(img_array)

    # Inference
    prediction, confidence, heatmap = predict_image_class(model=model_cell,
                                                          image=img_array,
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

@app.route('/singlecell', methods=['POST'])
def singlecell_file():
    """
    Handles the single cell prediction
    """
    
    global model_cell_name
    global model_cell
    
    # Get the uploaded image file
    image = request.files['image']

    # Convert image to base64 string
    image_string = base64.b64encode(image.read()).decode('utf-8')

    # Convert to PIL format and compute scale factor
    image = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(image))
    min_width = max(image.size[0], 128.)
    scale_fac = image.size[0] / min(1536., min_width)
    
    # Setting image type for prediction and rendering and load the model if required
    image_type = 'Single cell'
    if model_cell == None:
        model_cell = load_model(model_cell_name, custom_objects={"class2_recall": class2_recall})
    
    # Preprocessing the image for prediciton
    image_resized = image.resize((128,128), resample=Image.Resampling.BILINEAR)
    img_array = img_to_array(image_resized)
    img_array = preprocess_input_vgg19(img_array)

    # Inference
    prediction, confidence, heatmap = predict_image_class(model=model_cell,
                                                          image=img_array,
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
                           image=image_string,
                           prediction=prediction,
                           confidence=confidence_string,
                           image_type=image_type,
                           img_scale=scale_fac,
                           heatmap=heatmap_b64)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
