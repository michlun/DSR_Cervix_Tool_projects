from flask import Flask, render_template, request
import base64
from PIL import Image
import io
from model_2_utils import predict_image_class

app = Flask(__name__)

SCALE = 2

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
        image_type = 'Whole slide'
        prediction, confidence = predict_image_class(model='models/conv1_192x256_lr001_1dense256.h5', image=image, image_type=image_type)
        confidence_string = str(round(confidence * 100, 2))

        # Render the results in a page along with the image
        return render_template('results.html',
                               prediction=prediction,
                               confidence=confidence_string,
                               image=image_string,
                               image_type=image_type,
                               img_scale=SCALE)

    return render_template('upload.html')


@app.route('/annotate', methods=['POST'])
def annotate_file():
    # Retrieve the annotated region from the form submission
    xstart = int(request.form['xstart']) * SCALE
    xend = int(request.form['xend']) * SCALE
    ystart = int(request.form['ystart']) * SCALE 
    yend = int(request.form['yend']) * SCALE
    image_base64 = request.form['image']
    print('Xstart', xstart, 'Xend', xend, 'Ystart', ystart, 'Yend', yend)

    # crop the image based on the annotated region
    image = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image))
    width, height = image.size
    print('Uncropped image size:', image.size)
    new_width = xend - xstart
    new_height = yend - ystart
    image = image.crop((xstart, ystart, xend, yend))
    # image = image.resize((new_width, new_height))
    print('Cropped image size:', image.size)
    
    # convert pillow image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


    # Prediction
    image_type = 'Single cell'
    prediction, confidence = predict_image_class(model='models/cell_conv1_aug_80x80_1dense128.h5', image=image, image_type=image_type)
    confidence_string = str(round(confidence * 100, 2))

    # Render the results in a page along with the image
    return render_template('results.html', 
                           image=image_base64,
                           prediction=prediction,
                           confidence=confidence_string,
                           image_type=image_type,
                           img_scale=SCALE)


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
