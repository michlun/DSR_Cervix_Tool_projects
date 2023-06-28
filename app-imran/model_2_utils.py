import tensorflow as tf
import numpy as np
from PIL import Image
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from matplotlib import cm

def predict_image_class(model, image, type=None, heatmap=False, saliency=False):
    
    if type == "whole":
        resolution = (192,256)
        if image.size != (256,192):
            image = image.resize((256,192), resample=Image.Resampling.BILINEAR)
    if type == "cell":
        resolution = (80,80)
    
    class_names = ['Dyskeratotic', 'Superficial-Intermediate', 'Koilocytotic', 'Metaplastic', 'Parabasal']
    model = tf.keras.models.load_model(model)
    
    # image = tf.keras.utils.load_img(path=image, target_size=resolution, interpolation='bilinear', keep_aspect_ratio=False)
    img_array = tf.keras.utils.img_to_array(image)/255.0
    img_array_batch = np.expand_dims(img_array,axis=0)
    
    pred = model.predict(x=img_array_batch, batch_size=1)
    predicted_class = pred.argmax()
    confidence = pred.reshape(5)[predicted_class]
    
    replace2linear = ReplaceToLinear()
    score = CategoricalScore(predicted_class)
    
    if heatmap:
        # Create Gradcam object
        gradcam = Gradcam(model,
                  model_modifier=replace2linear,
                  clone=True)
        # Generate heatmap with GradCAM
        cam = gradcam(score,
              img_array,
              penultimate_layer=-1)
        heatmap = np.uint8(255 - (cm.jet(cam[0])[..., :3] * 255))
        return class_names[predicted_class], confidence, heatmap
    elif saliency:
        # Create Saliency object.
        saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)
        # Generate saliency map
        saliency_map = saliency(score, img_array) # , smooth_samples=20, smooth_noise=0.20)
        return class_names[predicted_class], confidence, saliency_map[0]
    else:
        return class_names[predicted_class], confidence
