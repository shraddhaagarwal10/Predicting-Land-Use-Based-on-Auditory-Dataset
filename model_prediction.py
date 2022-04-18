from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

import os
import numpy as np
from tensorflow import keras


test = ImageDataGenerator()

test_data = test.flow_from_directory("./image_datasets/test", shuffle = True, seed = 42, target_size=(512, 316))

model = keras.models.load_model("./model_save_state/model2")

final_location = ["Commercial_test", "Industrial_test", "Residential_test"]

count = 0
total_samples = 0

for area in final_location:
    dir_path = './image_datasets/test'+ "/" + area 
    
    total_samples += len(os.listdir(dir_path))

    for i in os.listdir(dir_path):
        img = image.load_img(dir_path + "//" + i, target_size = (316, 316))
        plt.imshow(img)

        img_arr = image.img_to_array(img)
        img_arr_expanded = np.expand_dims(img_arr, axis=0)
        images =  np.vstack([img_arr_expanded])

        model.predict(images)
        val = model.predict(images)
        if val[0][0] == 1:
            result = "Commercial"
            if result == area:
                count += 1
                print("Correct_prediction")
            else:
                print("Incorrect_prediction")

        elif val[0][1] == 1:
            result = "Industrial"
            if result == area:
                count += 1
                print("Correct_prediction")
            else:
                print("Incorrect_prediction")

        elif val[0][2] == 1:
            result = "Residential"
            if result == area:
                count += 1
                print("Correct_prediction")
            else:
                print("Incorrect_prediction")
            
       
print("\n\n")       
print("Accuracy of the model: ", (count/total_samples))