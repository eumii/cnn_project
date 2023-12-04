import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title=None, page_icon=None, layout="wide") # to use the whole page of the streamlit app 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


col1, col2 = st.columns([2,2])

# Convert the numpy array to a string
def array_to_string(arr):
    return ','.join(map(str, arr.flatten()))

# Convert the string back to a numpy array
def string_to_array(string, shape):
    flattened_arr = np.fromstring(string, dtype=int, sep=',')
    return flattened_arr.reshape(shape)


with st.sidebar:
    st.image('https://imgs.search.brave.com/3-KD5HmYh0GUizcofHFl9i8QCJ76VMnTzbl42yTU744/rs:fit:500:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMudGVjaGhpdmUu/Y29tL2ltYWdlcy9h/cnRpY2xlLzIwMTUv/MDgvdGhpbmtzdG9j/a3Bob3Rvcy00OTcy/NzQyMTEtMTAwNjA5/NjExLWxhcmdlLmpw/Zz9hdXRvPXdlYnAm/cXVhbGl0eT04NSw3/MA')
    st.title('Image Classification App with CNN')
    choice = st.radio('Navigation', ['Search & image selection', 'Our Model'])
    st.info('This app was build to classifiy movies images in genres using streamlit and CNN algorithms')

# Load the movie dataset
@st.cache_data
def load_data():
    data = pd.read_csv('df_movies_10k.csv')
    data["genre"] = data["genre"].apply(ast.literal_eval)
    return data

movies = load_data()

if choice == 'Search & image selection':
    if 'img_array' not in st.session_state:
        st.session_state.img_array = 0
    elif 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = 0
    elif 'img' not in st.session_state:
        st.session_state.img = 0
    elif 'class_name' not in st.session_state:
        st.session_state.class_name = 0
    elif 'class_pred_name' not in st.session_state:
        st.session_state.class_pred_name = 0

    with col1:
        # Add a title and a search bar
        st.title('Movie Search App Classification')
        user_input = st.text_input('Search for a movie title')
        # Filter the dataset based on user input
        filtered_movies = movies[movies['title'].str.contains(user_input, case=False)]
        # Display the search results
        if user_input:
            st.subheader('Search results:')
            if filtered_movies.empty:
                st.write('No movies found. Please try another search...')
                col1.error('you have not picked a movie yet !')
            else:
                # st.write(filtered_movies)
                col1.success('you picked a movie')
                # Show the cards
                N_cards_per_row = 4
                if user_input:
                    for n_row, row in filtered_movies.reset_index().iterrows():
                        i = n_row%N_cards_per_row
                        if i==0:
                            st.write("---")
                            cols = st.columns(N_cards_per_row, gap="small")
                        # draw the card
                        with cols[n_row%N_cards_per_row]:
                            st.caption(f"**{row['title']}** \n*{row['genre_id']}* ")
                            base_url = "https://image.tmdb.org/t/p/w500/"
                            poster_path = base_url+row['poster_path']
                            st.image(poster_path, caption='Movie Poster', use_column_width=True)
                            st.markdown(f"*{poster_path}*")
                
    with col2:        
        st.title('Upload image for modeling')
            #uploading the image
        uploaded_file  = st.file_uploader('upload image here ...', type = ["jpg", "jpeg", "png"])
        
        
        if uploaded_file is not None:
            #saving the image for sessions
            st.session_state.uploaded_file = uploaded_file
                # Display the uploaded image resized
            # new_size = st.slider(224, 10 , 1000, 100)
            # resized_img = uploaded_file.resize(width=new_size, height=new_size)
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

             # Load and preprocess the image
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            #save this variable for the next session state
            st.session_state.img_array = img_array
            #save this variable for the next session state
            st.session_state.img = img
           
                

if choice == 'Our Model':
    
    with col1:
        st.title('Our Predictions using our model')
        #loading the model saved 
        model_vgg16 = tf.keras.models.load_model('modelTrain_drop02.h5')
        img = st.session_state.img
        # Load the VGG16 model
        image2 = img.convert("RGB")
        image2 = image2.resize((224, 224))  # Redimensionner toutes les images à la même taille
        image2 = np.array(image2, dtype="float32")
        image_array = tf.keras.applications.vgg16.preprocess_input(image2)
        image_array = np.expand_dims(image_array, axis=0)
        # Make predictions based on the uploaded image
        predictions = model_vgg16.predict(image_array)
        decoded_predictions = np.argmax(predictions, axis=1)


        mlb = MultiLabelBinarizer()
        mlb.fit(movies["genre"])
        # st.write(mlb.classes_)
        tt = np.expand_dims(predictions[0] > 0.5, axis=0)
        class_pred_name = mlb.inverse_transform(tt)
        #save this variable for the next session state
        st.session_state.class_name = mlb.classes_

        st.write(f'Predicted Classes: {class_pred_name}')
        st.write(f'Number of classes: {len(class_pred_name[0])}')
        st.session_state.class_pred_name = class_pred_name
    
    with col2:
        st.write('the true class are:')
        st.markdown(movies[movies["poster_path"] == f"/{st.session_state.uploaded_file.name}"]["genre"].values[0])
        st.image(st.session_state.uploaded_file)








