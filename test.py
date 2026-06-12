import keras.models
from keras import Model
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import ETD_net


from Optimize import dispose_img, filter_continuous_columns
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)
import warnings
warnings.filterwarnings("ignore")
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

forgery_attention_model = ETD_net.model()

# forgery_attention_model.summary()

forgery_attention_model.load_weights('epoch-27.h5')


for k in tqdm(range(1, 501), position=0):

    filename = rf'your mel path\{k}_mel.png'

    if not os.path.exists(filename):

        continue

    try:
        img = image.load_img(filename, target_size=(480, 640))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        forgery_attention_intermediate_output = forgery_attention_model.predict(img_tensor)

        mask = forgery_attention_intermediate_output
        test = mask[0, :, :, 0]
        test[test > 0.95] = 255
        test[test < 0.95] = 0
        test = dispose_img(test)
        test = filter_continuous_columns(test)
        plt.imshow(test, cmap='gray')
        plt.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

        new_file_name = f"{k}_res.png"
        new_save_path = os.path.join(r'your result path', new_file_name)

        plt.savefig(new_save_path)
        plt.close()

    except Exception as e:

        print(f"Failed to process file {filename}: {str(e)}")

        continue
