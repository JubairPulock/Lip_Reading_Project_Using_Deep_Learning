import streamlit as st
import os
import imageio
import subprocess
import tensorflow as tf
from utils import load_data_by_path, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip Reader')
    st.info('Hello everyone. I am Jubair Mahmud Pulock, and this is my CSE499 project.')

st.title('Lip Reading Full Stack App')
options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        video_path = 'test_video.mp4'
        video = imageio.get_reader(video_path, 'ffmpeg')
        video_bytes = video.get_meta_data()['fps']
        st.video(video_path)

    with col2:
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        frames, _ = load_data_by_path(tf.constant(file_path))
        yhat = model.predict(tf.expand_dims(frames, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decoding the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
