import numpy as np
from keras import optimizers
from vggmodel2 import NetworkModel
from keras.preprocessing import image as image_utils
# 15 is Next
# 16 is Previous
# 17 is start
# 18 is stop
class_names=["sila", "almusal", "paa", "baunan", "tigre", "serbisyo", "transportasyon", "ekskursiyon", "sobre", "miyembro", "abstrak", "eksplorasyon"]
weights_path=r"D:\Thesis_App\Model\td_vgg_4_cp.hdf5"

# Build VGG model
m = NetworkModel()
model = m.time_distributed_vgg()
# Load weights for model
model.load_weights(weights_path)

def predict_by_model(path):
    # example path = 'val/16/M01_words_06_01result.jpg'

    print("[INFO] loading and preprocessing image...")
    input_image = load_and_prcoess_image(path)
    prediction = model.predict(input_image)
    prediction_class = np.argmax(prediction, axis=1)

    if(is_confidence_too_low(prediction)):
        print("Can you say again? Please")
        write_to_txt("result_lip/text.txt", "Can you say again? Please")

    else:
        print(class_names[prediction_class[0]])
        print(prediction_class[0]+1)
        print(prediction[0])
        write_to_txt("result_lip/text.txt", class_names[prediction_class[0]])

    # ID of Good Bye is 5
    if(prediction_class[0]+1==1):
        return 0
    else:
        return 1

def load_and_prcoess_image(path):
    image = image_utils.load_img(path, target_size=(50, 48))
    image = image_utils.img_to_array(image)
    input_image = np.expand_dims(image, axis=0)/255
    return input_image

def is_confidence_too_low(prediction):
    prediction_class = np.argmax(prediction, axis=1)
    return prediction[0][prediction_class[0]]<0.5

def write_to_txt(name, words):
    with open(name, "w") as text_file:
        text_file.write(words)  
