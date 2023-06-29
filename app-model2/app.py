from flask import Flask, render_template, request
import base64
from PIL import Image
import io
from model_2_utils import predict_image_class, class_recall, class2_recall
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load prediction models
model_whole = 'models/conv1_192x256_lr001_1dense256.h5'
model_whole = load_model(model_whole)
model_cell = 'models/cell_conv1_aug_80x80_1dense128.h5'
model_cell = load_model(model_cell, custom_objects={"class2_recall": class2_recall})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        # Get the uploaded image file
        image = request.files['image']
        
        # convert image to base64 string
        image_string = base64.b64encode(image.read()).decode('utf-8')
        
        # convert to PIL format and compute scale factor
        image = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image))
        scale_fac = image.size[0]/1024.
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
    app.run(debug=True)
