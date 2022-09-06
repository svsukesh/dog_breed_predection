#Importing required libraries
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

#loading the model
model= load_model('dog_breed.h5')

#selecting the required breed names
class_names = ['scottish_deerhound', 'maltese_dog','afghan_hound' ]

#Setting the title of the app
st.title('Dog Breed Predection')
st.markdown('Upload an image')

#uploading the file
image_file =st.file_uploader('Seclect an image to upload....', type='PNG')
submit = st.button('Predict')

#On predict buction click
if submit:

    if image_file is not None:

        #converting image to byte code
        byte_file = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(byte_file, 1)

        #displaying the picture
        st.image(opencv_image, channels='RGB')

        #reszing the image
        opencv_image = cv2.resize(opencv_image, (224, 224))

        #Concerting the image to 4 dimentions
        opencv_image.shape = (1,224,224,3)
 
        Y_pred = model.predict(opencv_image)

        st.title(str('The dog breed is '+class_names[np.argmax(Y_pred)]))


