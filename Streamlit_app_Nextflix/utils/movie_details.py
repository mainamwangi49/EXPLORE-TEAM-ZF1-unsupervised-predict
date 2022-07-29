import numpy as np
import pandas as pd
from IPython.display import IFrame
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup
import requests,io
from io import BytesIO
import PIL.Image
from urllib.request import urlopen



def movie_poster_fetcher(df):
    ## Display Movie Poster
    url_data = requests.get(df).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_dp = s_data.find("meta", property="og:image")
    movie_poster_link = imdb_dp.attrs['content']
    u = urlopen(movie_poster_link)
    #---------------------------------------------------
    byteImgIO = io.BytesIO()
    byteImg = Image.open(u)
    byteImg.save(byteImgIO, "PNG")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read() 
    
    dataBytesIO = io.BytesIO(byteImg)
    image = Image.open(dataBytesIO)
    #---------------------------------------------------
    
    image = image.resize((158, 301), )
    st.image(image, use_column_width=True)
    
    
    
def get_movie_info(df):
    url_data = requests.get(df).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_content = s_data.find("meta", property="og:description")
    movie_descr = imdb_content.attrs['content']
    movie_descr = str(movie_descr).split('.')
    movie_director = movie_descr[0]
    movie_cast = str(movie_descr[1])
    movie_story = str(movie_descr[2]).strip()+'.'
    
    director = movie_director.split(":")[-1]
    title = ':'.join(movie_director.split(":")[0:-1])
    cast = movie_cast
    story = movie_story

    return {'Director': director, 
           'Title': title,
           'Cast': cast,
           'Story': story}