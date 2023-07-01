import numpy as np
from PIL import Image
from tensorflow import keras
from keras.utils import img_to_array
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from matplotlib import cm
import keras.backend as K


def class_recall(y_true, y_pred, c):
    y_true = K.flatten(y_true)
    pred_c = K.cast(K.equal(K.argmax(y_pred, axis=-1), c), K.floatx())
    true_c = K.cast(K.equal(y_true, c), K.floatx())
    true_positives = K.sum(pred_c * true_c)
    possible_postives = K.sum(true_c)
    return true_positives / (possible_postives + K.epsilon())


def class2_recall(y_true, y_pred):
    return class_recall(y_true, y_pred, 2)


def predict_image_class(model, image, image_type=None, gradcam_map=False, saliency_map=False):

    if image_type == "Whole slide":
        resolution = (192,256)
        if image.size != (256,192):
            image = image.resize((256,192), resample=Image.Resampling.BILINEAR)
    if image_type == "Single cell":
        resolution = (80,80)
        if image.size != (80,80):
             image = image.resize((80,80), resample=Image.Resampling.BILINEAR)

    class_names = ['Dyskeratotic', 'Superficial-Intermediate', 'Koilocytotic', 'Metaplastic', 'Parabasal']

    # image = tf.keras.utils.load_img(
    #       path=image, target_size=resolution, interpolation='bilinear', keep_aspect_ratio=False)
    img_array = img_to_array(image)/255.0
    img_array_batch = np.expand_dims(img_array,axis=0)

    pred = model.predict(x=img_array_batch, batch_size=1)
    predicted_class = pred.argmax()
    confidence = pred.reshape(5)[predicted_class]

    replace2linear = ReplaceToLinear()
    score = CategoricalScore(predicted_class)

    if gradcam_map:
        # Create Gradcam++ object
        gradcam = GradcamPlusPlus(model,
                  model_modifier=replace2linear,
                  clone=True)
        # Generate heatmap with GradCAM++
        cam = gradcam(score,
              img_array,
              penultimate_layer=-1)
        heatmap = np.uint8((cm.jet(cam[0])[..., :3] * 255))
        return class_names[predicted_class], confidence, heatmap
    elif saliency_map:
        # Create Saliency object.
        saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)
        # Generate saliency map
        saliency_map = saliency(score, img_array) # , smooth_samples=20, smooth_noise=0.20)
        return class_names[predicted_class], confidence, saliency_map[0]
    else:
        return class_names[predicted_class], confidence