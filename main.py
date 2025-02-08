import streamlit as st
import tensorflow as tf
import numpy as np
st.title('Plant Disease Detection')

#Input for model
def model_predictions(test_image):
    model = tf.keras.models.load_model('NPDDCustom_model.keras')
    input_image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = np.array([tf.keras.preprocessing.image.img_to_array(input_image)])
    predictions = model.predict(input_arr)
    return_index =  np.argmax(predictions)
    return return_index

#Sidebar

st.sidebar.title('Choose Image')
image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])

if image_file is not None:
    returned = model_predictions(image_file)
    st.image(image_file)
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
    st.write(class_names[returned])