from flask import Flask, render_template, request
import base64
from PIL import Image
import io
from model_2_utils import predict_image_class

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image file
        image = request.files['image']
        
        # convert image to base64 string
        image_string = base64.b64encode(image.read()).decode('utf-8')
        
        # convert to PIL format
        image = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image))

        # inference placeholder (pass the image to the model and predict)
        prediction, confidence = predict_image_class(model='models/conv1_192x256_lr001_1dense256.h5', image=image, type='whole')
        confidence_string = str(round(confidence * 100, 2))

        # Render the results in a page along with the image
        return render_template('results.html', prediction=prediction, confidence=confidence_string, image=image_string)

    return render_template('upload.html')


@app.route('/annotate', methods=['POST'])
def annotate_file():
    # Retrieve the annotated region from the form submission
    xstart = request.form['xstart']
    xend = request.form['xend']
    ystart = request.form['ystart']
    yend = request.form['yend']
    image_base64 = request.form['image']


    # crop the image based on the annotated region
    image = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image))
    width, height = image.size
    new_width = int(xend) - int(xstart)
    new_height = int(yend) - int(ystart)
    image = image.crop((int(xstart), int(ystart), int(xend), int(yend)))
    image = image.resize((new_width, new_height))

    # convert pillow image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


    # Process the annotated region and calculate the new width and height
    prediction, confidence = predict_image_class(model='models/...', image=image, type='cell')

    # Render the results in a page along with the image
    return render_template('results.html', width=new_width, height=new_height, image=image_base64, prediction=prediction)


def run_inference(image_base64):
    """
    Placeholder for running the inference on the image.
    Returns the aspect ratio of the image.
    """
    # convert base64 string to image
    image = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image))

    # Get the aspect ratio of the image
    width, height = image.size
    aspect_ratio = width / height

    # Implement your code
    prediction = aspect_ratio
    return prediction



if __name__ == '__main__':
    app.run(debug=True)
