"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import random
from ssl import Options
import streamlit as st
from turtle import color, width
import streamlit as st
import streamlit.components.v1 as stc
from streamlit_option_menu import option_menu
import joblib, os
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
import time
from PIL import Image
import pickle as pkle
import os.path


# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles, load_most_recent_movies, load_year_data
from utils.data_loader import load_genre_data, load_director_data, load_merged_data, load_ratings_data
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from utils.movie_details import movie_poster_fetcher, get_movie_info

import base64

favicon = Image.open('resources/imgs/nextflix_icon.png')
st.set_page_config(page_title="NextFlix", page_icon=favicon)


#------------------------------------------------------------------------------------------------------------
# Adding footer to the StreamLit App
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 50px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height= 0.2, #"auto",
        opacity=1             
    )

    style_hr = styles(
        display="border",
        margin=px(5, 5, "auto", "auto"),
        border_style="none",
        border_width=px(0.5),
        color = 'rgba(0,0,0,.5)'
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        # "Made in ",
        # image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
        #       width=px(25), height=px(25)),
        " Powered ❤️ by ",
        link("https://twitter.com/ChristianKlose3", "@ZF1 Consult"),
        br(),
        # link("https://buymeacoffee.com/chrischross", image('https://i.imgur.com/thJhzOO.png')),
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()

# footer="""<style>
# a:link , a:visited{
# color: #ED2E38;
# background-color: transparent;
# text-decoration: underline;
# }

# a:hover,  a:active {
# color: white;
# background-color: transparent;
# text-decoration: underline;
# }

# .footer {
# position: fixed;
# left: 0;
# bottom: 0;
# width: 100%;
# background-color: rgba(0,0,0,.5);
# color: white;
# text-align: center;
# }
# </style>
# <div class="footer">
# <p>Developed with ❤ by <a style='display:' href="https://www.heflin.dev/" target="_blank">@DeftAlpGlobal</a></p>
# </div>
# """
# st.markdown(footer,unsafe_allow_html=True)
    #-----------------------------------------------------------------------------------------------------------



@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('resources/imgs/background.png')

# Data Loading
most_recent = load_most_recent_movies('resources/data/most_recent.csv')
sample_recent = most_recent.head(100).sample(3)
year_df = load_year_data('resources/data/merged_data.csv')
genre_df = load_genre_data('resources/data/merged_data.csv')
director_df = load_director_data('resources/data/merged_data.csv')
title_list = load_movie_titles('resources/data/movies.csv')
selected_data = load_merged_data('resources/data/merged_data.csv')
sorted_ratings = load_ratings_data('resources/data/ratings.csv')


def local_button_css(file_name):
                    with open(file_name) as f:
                        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
                
local_button_css("utils/button_style.css")

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# local_css("utils/style.css")


# App declaration
def main():
    

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    # page_options = ["Home", "Recommender System","Solution Overview"]

    # Design horizontal bar
    # menu = ["Home", "EDA", "Prediction", "About"]
    page_options = ["Recommender System", "Movies", "EDA", "About"]
    selection = option_menu( menu_title=None,
                            options=page_options,
                            icons=["house", "camera-reels", "graph-up", "file-person"],
                            orientation='horizontal',
                            styles={
                                        "container": {"padding": "0!important", "background-color": "#0098DA"},
                                        "icon": {"color": "black", "font-size": "15px",  },
                                        "nav-link": {
                                            "font-size": "15px",
                                            "text-align": "center",
                                            "margin": "5px",
                                            "--hover-color": "#eee",
                                            "color": "white"
                                        },
                                        "nav-link-selected": {"background-color": "white", "color": "#0098DA"},
                                    },
        )    

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    # page_selection = st.sidebar.selectbox("Choose Option", page_options)
    page_selection = selection
    if page_selection == "Recommender System":
        st.markdown("<h2 style='text-align: center; color: white;'>NextFlix, Life!, we bring to you!</h2>", unsafe_allow_html=True)
        #st.markdown("<h3 style='text-align: center; color: #0098DA;'> Top Rated Movies!</h3>", 
                    # unsafe_allow_html=True)
        
        sys = st.radio("Select an algorithm",
            ('Content Based Filtering',
                'Collaborative Based Filtering'))

            # selected_title = [title1, title2, title3, title4, title5]
            
            # User-based preferences
        st.write('#### Click the button below for more recommendation')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200]) # random.choice(selected_title)
        movie_2 = st.selectbox('Second Option',title_list[25055:25255]) # random.choice(selected_title)
        movie_3 = st.selectbox('Third Option',title_list[21100:21200]) # random.choice(selected_title)
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                            We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                        top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                            We'll need to fix it!")
        
        # picture1, picture2, picture3 = st.columns(3)
        # with picture1:
        #     picture1 = movie_poster_fetcher(sample_recent['url'].iloc[0]) 
        #     with st.expander("About Movie"):
        #         desc = get_movie_info(sample_recent['url'].iloc[0])
        #         st.markdown(f"""
        #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #             <b style='color: #0098DA'>Year: </b>{int(sample_recent['year'].iloc[0])} Movie<br><br>
        #             <b style='color: #0098DA'>Director: </b>{desc['Director']}<br><br>
        #             <b style='color: #0098DA'>Title: </b>{desc['Title']}<br><br>
        #             <b style='color: #0098DA'>Cast: </b>{desc['Cast']}<br><br>
        #             <b style='color: #0098DA'>Story: </b>{desc['Story']}<br><br>
        #             <b style='color: #0098DA'>Watch Trailer: 👇 </b><br>
        #             <a href={sample_recent['url'].iloc[0]}>Click to watch trailer</a>
        #             </p>""", unsafe_allow_html=True)

        # with picture2:
        #     picture2 = movie_poster_fetcher(sample_recent['url'].iloc[1])
        #     with st.expander("About Movie"):
        #         desc = get_movie_info(sample_recent['url'].iloc[1])
        #         st.markdown(f"""
        #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #             <b style='color: #0098DA'>Year: </b>{int(sample_recent['year'].iloc[1])} Movie<br><br>
        #             <b style='color: #0098DA'>Director: </b>{desc['Director']}<br><br>
        #             <b style='color: #0098DA'>Title: </b>{desc['Title']}<br><br>
        #             <b style='color: #0098DA'>Cast: </b>{desc['Cast']}<br><br>
        #             <b style='color: #0098DA'>Story: </b>{desc['Story']}<br><br>
        #             <b style='color: #0098DA'>Watch Trailer: 👇 </b><br>
        #             <a href={sample_recent['url'].iloc[1]}>Click to watch trailer</a>
        #             </p>""", unsafe_allow_html=True)

        
        # with picture3:
        #     picture3 =  movie_poster_fetcher(sample_recent['url'].iloc[2]) 
        #     with st.expander("About Movie"):
        #         desc = get_movie_info(sample_recent['url'].iloc[2])
        #         st.markdown(f"""
        #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #             <b style='color: #0098DA'>Year: </b>{int(sample_recent['year'].iloc[2])} Movie<br><br>
        #             <b style='color: #0098DA'>Director: </b>{desc['Director']}<br><br>
        #             <b style='color: #0098DA'>Title: </b>{desc['Title']}<br><br>
        #             <b style='color: #0098DA'>Cast: </b>{desc['Cast']}<br><br>
        #             <b style='color: #0098DA'>Story: </b>{desc['Story']}<br><br>
        #             <b style='color: #0098DA'>Watch Trailer: 👇 </b><br>
        #             <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #             </p>""", unsafe_allow_html=True)

            
    
    
    # if page_selection == "Recommender System":
    if page_selection == "Movies":

        st.markdown("<h2 style='text-align: center; color: white;'>What movie are you watching today?</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: white;'>Choose one or more of the options below for the best movie </h4>", unsafe_allow_html=True)
        
        choice_collection = []

        option1, option2, option3 = st.columns(3)
        with option1:
            movie_year = st.checkbox("Choose Movie Year")
            if movie_year:
                selected_year = st.selectbox(
                    "Select a year", 
                    list(year_df))
                if selected_year:
                    choice_one = choice_collection.append(selected_year)
                else:
                    pass

        with option2:
            genre = st.checkbox("Choose Genre")
            if genre:
                selected_genre = st.selectbox(
                    "Select genre", 
                    list(genre_df))
                if selected_genre:
                    choice_two = choice_collection.append(selected_genre)
                else:
                    pass
                
        with option3:
            director = st.checkbox("Choose a Director")
            if director:
                selected_director = st.selectbox(
                    "Select director", 
                    director_df)
                if selected_director:
                    choice_three = choice_collection.append(selected_director)
                else:
                    pass
        button1, button2, button3 = st.columns(3)

        with button1:
            pass

        with button3:
            pass
        
        with button2:
            button_pressed = button2.button('Search for Movies')

        
        if button_pressed:
            selected_movieid = np.where((selected_data['year'] == choice_collection[0]) | ((selected_data['genre'] == choice_collection[1])))
            sliced_selected_movies = selected_data.iloc[selected_movieid]
        
            suggested_head = sliced_selected_movies.sort_values('year', ascending=False)#.head(50)
            suggested = suggested_head.sample(5)

            
            suggestion1, suggestion2, suggestion3, suggestion4, suggestion5 = st.columns(5)
            
            if "load_state" not in st.session_state:
                st.session_state.load_state = False                 

            with suggestion1:
                suggestion1 = movie_poster_fetcher(suggested['url'].iloc[0])
                if 'url1' not in st.session_state:
                    st.session_state['url1'] = suggested['url'].iloc[0]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[0])
                    title1 = desc['Title']
                    st.markdown(f"""
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[0])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion2:
                suggestion2 = movie_poster_fetcher(suggested['url'].iloc[1])
                if 'url2' not in st.session_state:
                    st.session_state['url2'] = suggested['url'].iloc[1]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[1])
                    title2 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[1])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion3:
                suggestion3 = movie_poster_fetcher(suggested['url'].iloc[2])
                if 'url3' not in st.session_state:
                    st.session_state['url3'] = suggested['url'].iloc[2]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[2])
                    title3 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[2])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion4:
                suggestion4 = movie_poster_fetcher(suggested['url'].iloc[3])
                if 'url4' not in st.session_state:
                    st.session_state['url4'] = suggested['url'].iloc[3]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[3])
                    title4 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[3])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion5:
                suggestion5 = movie_poster_fetcher(suggested['url'].iloc[4])
                if 'url5' not in st.session_state:
                    st.session_state['url5'] = suggested['url'].iloc[4]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[4])
                    title5 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[4])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)
            
            st.write("""
            <h6 style='text-align: center; color: white'>
            Congratulations! Here are some recommendations for you: settle in, unwind, and have fun.</h6>""", unsafe_allow_html=True)
            st.balloons()
        # with st.sidebar:
        #     logo = Image.open("resources/imgs/nextflix_logo2.png")
        #     st.image(logo, width =None, use_column_width='False')

        #     st.markdown(" ")
        #     st.markdown(" ")

        #     select_option = option_menu("Have you seen any of the suggested movies above?", ['Content-Based', 'Collaborative-based'], 
        #             icons=['people-fill', 'person-circle'], menu_icon="file-person", default_index=0)

        # if select_option == 'Content-Based':
        #     st.markdown("<h2 style='text-align: center; color: white;'>What movie are you watching today?</h2>", unsafe_allow_html=True)
        #     st.markdown("<h4 style='text-align: center; color: white;'>Choose one or more of the options below for the best movie </h4>", unsafe_allow_html=True)
            
        #     choice_collection = []

        #     option1, option2, option3 = st.columns(3)
        #     with option1:
        #         movie_year = st.checkbox("Choose Movie Year")
        #         if movie_year:
        #             # st.icon("search")
        #             # selected_year = st.text_input("", "Search...")
        #             selected_year = st.selectbox(
        #                 "Select a year", 
        #                 list(year_df))
        #             if selected_year:
        #                 choice_one = choice_collection.append(selected_year)
        #             else:
        #                 pass

        #     with option2:
        #         genre = st.checkbox("Choose Genre")
        #         if genre:
        #             selected_genre = st.selectbox(
        #                 "Select genre", 
        #                 list(genre_df))
        #             if selected_genre:
        #                 choice_two = choice_collection.append(selected_genre)
        #             else:
        #                 pass
                    
        #     with option3:
        #         director = st.checkbox("Choose a Director")
        #         if director:
        #             selected_director = st.selectbox(
        #                 "Select director", 
        #                 director_df)
        #             if selected_director:
        #                 choice_three = choice_collection.append(selected_director)
        #             else:
        #                 pass
        #     button1, button2, button3 = st.columns(3)

        #     with button1:
        #         pass

        #     with button3:
        #         pass
            
        #     with button2:
        #         button_pressed = button2.button('Search for Movies')

            
        #     if button_pressed:
        #         selected_movieid = np.where((selected_data['year'] == choice_collection[0]) | ((selected_data['genre'] == choice_collection[1])))
        #         sliced_selected_movies = selected_data.iloc[selected_movieid]
            
        #         suggested_head = sliced_selected_movies.sort_values('year', ascending=False)#.head(50)
        #         suggested = suggested_head.sample(5)
   
                
        #         suggestion1, suggestion2, suggestion3, suggestion4, suggestion5 = st.columns(5)
                
        #         if "load_state" not in st.session_state:
        #             st.session_state.load_state = False                 

        #         with suggestion1:
        #             suggestion1 = movie_poster_fetcher(suggested['url'].iloc[0])
        #             if 'url1' not in st.session_state:
        #                 st.session_state['url1'] = suggested['url'].iloc[0]

        #             with st.expander("About Movie"):
        #                 desc = get_movie_info(suggested['url'].iloc[0])
        #                 title1 = desc['Title']
        #                 st.markdown(f"""
        #                     <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                     <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[0])} Movie<br><br>
        #                     <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                     <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                     <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                     <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                     <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                     <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                     </p>""", unsafe_allow_html=True)

        #         with suggestion2:
        #             suggestion2 = movie_poster_fetcher(suggested['url'].iloc[1])
        #             if 'url2' not in st.session_state:
        #                 st.session_state['url2'] = suggested['url'].iloc[1]

        #             with st.expander("About Movie"):
        #                 desc = get_movie_info(suggested['url'].iloc[1])
        #                 title2 = desc['Title']
        #                 st.markdown(f"""
        #                     <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                     <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[1])} Movie<br><br>
        #                     <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                     <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                     <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                     <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                     <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                     <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                     </p>""", unsafe_allow_html=True)

        #         with suggestion3:
        #             suggestion3 = movie_poster_fetcher(suggested['url'].iloc[2])
        #             if 'url3' not in st.session_state:
        #                 st.session_state['url3'] = suggested['url'].iloc[2]

        #             with st.expander("About Movie"):
        #                 desc = get_movie_info(suggested['url'].iloc[2])
        #                 title3 = desc['Title']
        #                 st.markdown(f"""
        #                     <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                     <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[2])} Movie<br><br>
        #                     <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                     <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                     <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                     <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                     <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                     <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                     </p>""", unsafe_allow_html=True)

        #         with suggestion4:
        #             suggestion4 = movie_poster_fetcher(suggested['url'].iloc[3])
        #             if 'url4' not in st.session_state:
        #                 st.session_state['url4'] = suggested['url'].iloc[3]

        #             with st.expander("About Movie"):
        #                 desc = get_movie_info(suggested['url'].iloc[3])
        #                 title4 = desc['Title']
        #                 st.markdown(f"""
        #                     <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                     <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[3])} Movie<br><br>
        #                     <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                     <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                     <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                     <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                     <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                     <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                     </p>""", unsafe_allow_html=True)

        #         with suggestion5:
        #             suggestion5 = movie_poster_fetcher(suggested['url'].iloc[4])
        #             if 'url5' not in st.session_state:
        #                 st.session_state['url5'] = suggested['url'].iloc[4]

        #             with st.expander("About Movie"):
        #                 desc = get_movie_info(suggested['url'].iloc[4])
        #                 title5 = desc['Title']
        #                 st.markdown(f"""
        #                     <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                     <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[4])} Movie<br><br>
        #                     <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                     <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                     <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                     <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                     <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                     <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                     </p>""", unsafe_allow_html=True)
                
        #         st.write("""
        #         <h6 style='text-align: center; color: white'>
        #         Congratulations! Here are some recommendations for you: settle in, unwind, and have fun.</h6>""", unsafe_allow_html=True)
        #         #st.balloons()
                
    
                    
        # else:
        #     st.write("""
        #         <h6 style='text-align: center; color: white'>
        #         Want more recommedations? </h6>""", unsafe_allow_html=True)
            
        #     sys = st.radio("Select an algorithm",
        #     ('Content Based Filtering',
        #         'Collaborative Based Filtering'))

        #     #selected_title = [title1, title2, title3, title4, title5]
        #     # User-based preferences
        #     st.write('### Enter Your Three Favorite Movies')
        #     movie_1 = st.selectbox('First Option',title_list[14930:15200]) #random.choice(selected_title) # 
        #     movie_2 = st.selectbox('Second Option',title_list[25055:25255]) #random.choice(selected_title) # 
        #     movie_3 = st.selectbox('Third Option',title_list[21100:21200]) #random.choice(selected_title) # 
        #     fav_movies = [movie_1 ,movie_2,movie_3]
            

        #     #Perform top-10 movie recommendation generation
        #     if sys == 'Content Based Filtering':
        #         if st.button("Recommend"):
        #             try:
        #                 with st.spinner('Crunching the numbers...'):
        #                     top_recommendations = content_model(movie_list=fav_movies,
        #                                                     top_n=10)

        #                 st.title("We think you'll like:")
        #                 for i,j in enumerate(top_recommendations):
        #                     st.subheader(str(i+1)+'. '+j)
        #             except:
        #                 st.error("Oops! Looks like this algorithm does't work.\
        #                         We'll need to fix it!")


        #     if sys == 'Collaborative Based Filtering':
        #         if st.button("Recommend"):
        #             try:
        #                 with st.spinner('Crunching the numbers...'):
        #                     top_recommendations = collab_model(movie_list=fav_movies,
        #                                                     top_n=10)
        #                 st.title("We think you'll like:")
        #                 for i,j in enumerate(top_recommendations):
        #                     st.subheader(str(i+1)+'. '+j)
        #             except:
        #                 st.error("Oops! Looks like this algorithm does't work.\
        #                         We'll need to fix it!")

            # poster1, poster2, poster3, poster4, poster5 = st.columns(5)               

            # with poster1:
            #     poster1 = movie_poster_fetcher(st.session_state['url1'])

            #     with st.expander("About Movie"):
            #         desc = get_movie_info(st.session_state['url1'])
            #         title1 = desc['Title']
            #         st.markdown(f"""
            #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>                
            #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
            #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
            #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
            #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}
            #             </p>""", unsafe_allow_html=True)

            # with poster2:
            #     poster2 = movie_poster_fetcher(st.session_state['url2'])
                
            #     with st.expander("About Movie"):
            #         desc = get_movie_info(st.session_state['url2'])
            #         title2 = desc['Title']
            #         st.markdown(f"""
            #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
            #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
            #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
            #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
            #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}
            #             </p>""", unsafe_allow_html=True)

            # with poster3:
            #     poster3 = movie_poster_fetcher(st.session_state['url3'])

            #     with st.expander("About Movie"):
            #         desc = get_movie_info(st.session_state['url3'])
            #         title3 = desc['Title']
            #         st.markdown(f"""
            #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
            #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
            #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
            #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
            #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}
            #             </p>""", unsafe_allow_html=True)

            # with poster4:
            #     poster4 = movie_poster_fetcher(st.session_state['url4'])

            #     with st.expander("About Movie"):
            #         desc = get_movie_info(st.session_state['url4'])
            #         title4 = desc['Title']
            #         st.markdown(f"""
            #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
            #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
            #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
            #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
            #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}
            #             </p>""", unsafe_allow_html=True)

            # with poster5:
            #     poster5 = movie_poster_fetcher(st.session_state['url5'])

            #     with st.expander("About Movie"):
            #         desc = get_movie_info(st.session_state['url5'])
            #         title5 = desc['Title']
            #         st.markdown(f"""
            #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
            #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
            #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
            #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
            #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}
            #             </p>""", unsafe_allow_html=True)

            #fav_movies
            
        # Header contents
        # st.write('# Movie Recommender Engine')
        

        # st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection

        # st.markdown("<h2 style='text-align: center; color: white;'>What movie are you watching today?</h2>", unsafe_allow_html=True)
        # st.markdown("<h4 style='text-align: center; color: white;'>Choose one or more of the options below for the best movie </h4>", unsafe_allow_html=True)

        # choice_collection = []

        # option1, option2, option3 = st.columns(3)
        # with option1:
        #     movie_year = st.checkbox("Choose Movie Year")
        #     if movie_year:
        #         # st.icon("search")
        #         # selected_year = st.text_input("", "Search...")
        #         selected_year = st.selectbox(
        #             "Select a year", 
        #             list(year_df))
        #         if selected_year:
        #             choice_one = choice_collection.append(selected_year)
        #         else:
        #             pass

        # with option2:
        #     genre = st.checkbox("Choose Genre")
        #     if genre:
        #         selected_genre = st.selectbox(
        #             "Select genre", 
        #             list(genre_df))
        #         if selected_genre:
        #             choice_two = choice_collection.append(selected_genre)
        #         else:
        #             pass
                
        # with option3:
        #     director = st.checkbox("Choose a Director")
        #     if director:
        #         selected_director = st.selectbox(
        #             "Select director", 
        #             director_df)
        #         if selected_director:
        #             choice_three = choice_collection.append(selected_director)
        #         else:
        #             pass
        # button1, button2, button3 = st.columns(3)

        # with button1:
        #     pass

        # with button3:
        #     pass
        
        # with button2:
        #     button_pressed = button2.button('Search for Movies')

        
        # if button_pressed:
        #     selected_movieid = np.where((selected_data['year'] == choice_collection[0]) | ((selected_data['genre'] == choice_collection[1])))
        #     sliced_selected_movies = selected_data.iloc[selected_movieid]
           
        #     suggested_head = sliced_selected_movies.sort_values('year', ascending=False)#.head(50)
        #     suggested = suggested_head.sample(5)
            
        #     suggestion1, suggestion2, suggestion3, suggestion4, suggestion5 = st.columns(5)
        #     with suggestion1:
        #         suggestion1 = movie_poster_fetcher(suggested['url'].iloc[0])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[0])
        #             title1 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[0])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion2:
        #         suggestion2 = movie_poster_fetcher(suggested['url'].iloc[1])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[1])
        #             title2 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[1])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion3:
        #         suggestion3 = movie_poster_fetcher(suggested['url'].iloc[2])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[2])
        #             title3 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[2])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion4:
        #         suggestion4 = movie_poster_fetcher(suggested['url'].iloc[3])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[3])
        #             title4 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[3])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion5:
        #         suggestion5 = movie_poster_fetcher(suggested['url'].iloc[4])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[4])
        #             title5 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[4])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #         with st.sidebar:
        #             logo = Image.open("resources/imgs/my_logo.png")
        #             st.image(logo, width =None, use_column_width='False')

        #             st.markdown(" ")
        #             st.markdown(" ")

        #             select_option = option_menu("Have you seen any of the suggested movies above?", ['Recommend', 'Yes', 'No'], 
        #                     icons=['graph-up-arrow', 'people-fill', 'person-circle'], menu_icon="file-person", default_index=0)
                
        #             if select_option == 'Recommend':
        #                 st.markdown("We'll recommend some more interesting movies for you, okay!")

        #             elif selected == 'Yes':
        #                 st.write("Let's recommend some more movies for you!")
        #                 sys = st.radio("Select an algorithm",
        #                 ('Content Based Filtering',
        #                     'Collaborative Based Filtering'))

        #                 selected_title = [title1, title2, title3, title4, title5]

        #             # User-based preferences
                #     st.write('### Enter Your Three Favorite Movies')
                #     movie_1 = random.choice(selected_title) # st.selectbox('Fisrt Option',title_list[14930:15200])
                #     movie_2 = random.choice(selected_title) # st.selectbox('Second Option',title_list[25055:25255])
                #     movie_3 = random.choice(selected_title) # st.selectbox('Third Option',title_list[21100:21200])
                #     fav_movies = [movie_1,movie_2,movie_3]

                #     # Perform top-10 movie recommendation generation
                #     if sys == 'Content Based Filtering':
                #         if st.button("Recommend"):
                #             try:
                #                 with st.spinner('Crunching the numbers...'):
                #                     top_recommendations = content_model(movie_list=fav_movies,
                #                                                         top_n=10)
                #                 st.title("We think you'll like:")
                #                 for i,j in enumerate(top_recommendations):
                #                     st.subheader(str(i+1)+'. '+j)
                #             except:
                #                 st.error("Oops! Looks like this algorithm does't work.\
                #                         We'll need to fix it!")


                #     if sys == 'Collaborative Based Filtering':
                #         if st.button("Recommend"):
                #             try:
                #                 with st.spinner('Crunching the numbers...'):
                #                     top_recommendations = collab_model(movie_list=fav_movies,
                #                                                     top_n=10)
                #                 st.title("We think you'll like:")
                #                 for i,j in enumerate(top_recommendations):
                #                     st.subheader(str(i+1)+'. '+j)
                #             except:
                #                 st.error("Oops! Looks like this algorithm does't work.\
                #                         We'll need to fix it!")

                # else: 
                #     st.write("We'll recommend something cool for you in a minute!")

        # st.sidebar.selectbox("Have you seen any of the suggested movies above?", ['---', 'Yes', 'No'])
        # option_picked = st.selectbox("Have you seen any of the suggested movies above?", ['---', 'Yes', 'No'])

        # if option_picked == 'Yes':
        #     [title1, title2, title3, title4, title5]
            
        #     recommended_1 = [title1, title2, title3, title4, title5]
        #     question, yes, no = st.columns([3, 1, 1])
        #     with question:
        #         st.markdown("Have you seen any of the suggested movies above?")
            
        #     with yes:
        #         pick_yes = st.checkbox("Yes")
        #         if pick_yes:
        #             st.markdown(recommended_1)
        #         else:
        #             pass
            
        #     # with none:
        #     #     pass

        #     with no:
        #         pick_no = st.checkbox("No")
        #         if pick_no:
        #             st.markdown(recommended_1)
        #         else:
        #             pass

            
            # button1, button2, button3 = st.columns(3)

            # with button1:
            #     pass

            # with button3:
            #     pass
            
            # with button2:
            #     button_pressed_2 = button2.button('Recommend Movives')


    
    # 



        # if button_pressed and choice_one:
        #     selected_movieid = np.where((selected_data['year'] == choice_collection[0]))
        #     sliced_selected_movies = selected_data.iloc[selected_movieid]

        #     suggested_head = sliced_selected_movies.sort_values('year', ascending=False)#.head(50)
        #     suggested = suggested_head.sample(5)
            
        #     suggestion1, suggestion2, suggestion3, suggestion4, suggestion5 = st.columns(5)
        #     with suggestion1:
        #         suggestion1 = movie_poster_fetcher(suggested['url'].iloc[0])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[0])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[0])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion2:
        #         suggestion2 = movie_poster_fetcher(suggested['url'].iloc[1])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[1])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[1])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion3:
        #         suggestion3 = movie_poster_fetcher(suggested['url'].iloc[2])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[2])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[2])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion4:
        #         suggestion4 = movie_poster_fetcher(suggested['url'].iloc[3])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[3])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[3])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion5:
        #         suggestion5 = movie_poster_fetcher(suggested['url'].iloc[4])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[4])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[4])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)
        
        # elif button_pressed and choice_two:
        #     selected_movieid = np.where((selected_data['year'] == choice_collection[0]) | ((selected_data['genre'] == choice_collection[1])))
        #     sliced_selected_movies = selected_data.iloc[selected_movieid]

        #     suggested_head = sliced_selected_movies.sort_values('year', ascending=False)#.head(50)
        #     suggested = suggested_head.sample(5)
            
        #     suggestion1, suggestion2, suggestion3, suggestion4, suggestion5 = st.columns(5)
        #     with suggestion1:
        #         suggestion1 = movie_poster_fetcher(suggested['url'].iloc[0])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[0])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[0])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion2:
        #         suggestion2 = movie_poster_fetcher(suggested['url'].iloc[1])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[1])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[1])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion3:
        #         suggestion3 = movie_poster_fetcher(suggested['url'].iloc[2])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[2])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[2])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion4:
        #         suggestion4 = movie_poster_fetcher(suggested['url'].iloc[3])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[3])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[3])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion5:
        #         suggestion5 = movie_poster_fetcher(suggested['url'].iloc[4])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[4])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[4])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)
        # elif button_pressed and choice_three:
        #     selected_movieid = np.where((selected_data['year'] == choice_collection[0]) | ((selected_data['genre'] == choice_collection[1]))
        #                                 | (selected_data['director'] == choice_collection[2]))
        #     sliced_selected_movies = selected_data.iloc[selected_movieid]
        #     # movieIds = [movieId for movieId in sliced_selected_movies['movieId'] if movieId in sorted_ratings['movieId']] 
        #     # suggested = selected_data[selected_data['movieId'].isin(movieIds)]

        #     suggested_head = sliced_selected_movies.sort_values('year', ascending=False)#.head(50)
        #     suggested = suggested_head.sample(5)
            
        #     suggestion1, suggestion2, suggestion3, suggestion4, suggestion5 = st.columns(5)
        #     with suggestion1:
        #         suggestion1 = movie_poster_fetcher(suggested['url'].iloc[0])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[0])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[0])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion2:
        #         suggestion2 = movie_poster_fetcher(suggested['url'].iloc[1])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[1])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[1])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion3:
        #         suggestion3 = movie_poster_fetcher(suggested['url'].iloc[2])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[2])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[2])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion4:
        #         suggestion4 = movie_poster_fetcher(suggested['url'].iloc[3])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[3])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[3])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

        #     with suggestion5:
        #         suggestion5 = movie_poster_fetcher(suggested['url'].iloc[4])
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(suggested['url'].iloc[4])
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[4])} Movie<br><br>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #                 <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #                 <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #                 </p>""", unsafe_allow_html=True)

    


    if page_selection == "EDA":
        st.subheader("Exploration Data Analysis")
        st.markdown('<iframe title="Recommender_Clara" width="700" height="400" src="https://app.powerbi.com/reportEmbed?reportId=b0d49333-defc-406c-9c49-53bc420e7ce9&autoAuth=true&ctid=2367c755-5748-4e40-9cb3-0a104c7d6d63&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWNhbmFkYS1jZW50cmFsLWItcHJpbWFyeS1yZWRpcmVjdC5hbmFseXNpcy53aW5kb3dzLm5ldC8ifQ%3D%3D" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)
    
    
    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    # if page_selection == "Solution Overview":
    if page_selection == "About":
        with st.sidebar:
            logo = Image.open("resources/imgs/nextflix_logo2.png")
            st.image(logo, width =None, use_column_width='False')

            st.markdown(" ")
            st.markdown(" ")

            selected = option_menu("About", ["Recommender System", 'About Team'], styles={"color": "#0098DA"}, 
                    icons=['graph-up-arrow', 'people-fill', 'person-circle'], 
                    menu_icon="file-person", 
                    default_index=0)
            

        if selected == 'Recommender System':
            st.header("**App Documentation**: Learn How to use the Recommender System")
            

            st.markdown("""<p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'> 
                    This app was primarily created for tweets expressing belief in climate change. There are four pages in the app which includes; `home page`, `predictions`, `Exploratory Data Analysis` and `About`.<br><br>
                    <b style='color: #ED2E38'>Home:</b> The home page is the app's landing page and includes a welcome message and a succinct summary of the app.<br><br>
                    <b style='color: #ED2E38'>EDA:</b> The EDA section, which stands for Explanatory Data Analysis, gives you the chance to explore your data. 
                    Based on the number of hash-tags and mentions in the tweet that have been gathered, it also displays graphs of various groups of 
                    words in the dataset, giving you a better understanding of the data you are working with.<br><br>
                    <b style='color: #ED2E38'>Prediction:</b> This page is where you use the main functionality of the app. It contains two subpages which are: `Single Text Prediction` and `Batch Prediction`<br><br>
                    <b style='color: #ED2E38'>Single Text Prediction:</b> You can predict the sentiment of a single tweet by typing or pasting it on the text prediction 
                    page. Enter any text in the textbox beneath the section, then click "Predict" to make a single tweet prediction.<br><br>
                    <b style='color: #ED2E38'>Batch Prediction:</b> You can make sentiment predictions for batches of tweets using this section. It can process multiple tweets in a batch from a `.csv` 
                    file with at least two columns named `message` and `tweetid` and categorize them into different tweet sentiment groups. To predict by file up, 
                    click on the `browse file` button to upload your file, then click on process to do prediction. A thorough output of the prediction will be provided, 
                    including a summary table and the number of tweets that were categorised under each sentiment class.<br><br>
                    <b style='color: #ED2E38'>About:</b> The About page also has two sub-pages;  `Documentation` and `About Team` page.<br><br>
                    <b style='color: #ED2E38'>Documentation:</b> This is the current page. It includes a detailed explanation of the app as well as usage guidelines on
                    how to use this app with ease.<br><br>
                    <b style='color: #ED2E38'>About Team:</b> This page gives you a brief summary of the experience of the team who built and manages the app.
                    </p>""", unsafe_allow_html=True)

        else:
            st.title("About Team")
            st.write("We work with seasoned professionals to give the best product experience")

            st.markdown(" ")
            daniel_pic = Image.open("resources/imgs/daniel (2).png")
            clara_pic = Image.open("resources/imgs/clara (2).jpg")         

            daniel, clara = st.columns(2)

            daniel.info("Founder/Growth Strategist")
            clara.info("Product Manager")
            
            with daniel:# ken's profile and picture
                st.header("Daniel")
                st.image(daniel_pic)

                with st.expander("Brief Bio"):
                    st.write("""
                    Founder of Neural Data Solution. Daniel has over 10 years experience as a Business Growth manager possessing additional
                    expertis in Product Develpoment. Proficient in facilitating business growth and enhancing market share of 
                    the company by leading in-depth market research and competitor analysis, liasing eith senior management and
                    conceptualizing new product development. 
                    
                    Highly skilled in functioning across multiple digital platforms and overseeing
                    product design to optimize process. Adept at building businesses and teams from scratch and spearheading Strategy, P&L 
                    Management, Marketing and Operations to lead data-driven decision making, render consumer impact analysis and achieve
                    astronomical growth with respect to profitability and customer acquisition.
                    """)

            with clara: #clara's profile and picture
                st.header("Clara")
                st.image(clara_pic)

                with st.expander("Brief Bio"):
                    st.write("""
                    Clara is a senior product manager with a background in user experience design and tons of experience in building
                    high quality softwares. She has experience with building high quality products and scaling them. Her attention to 
                    details is crucial as it has helped to work through models, visualizations, prototypes, requirements and manage across
                    functional team. 
                    
                    She works consistently with Data Scientists, Data Engineers, creatiives and other business-oriented 
                    people. She has gathered experience in data analytics, engineering, entrepreneurship, conversion optimization, internet 
                    marketing and UX. Using that experience, she has developed a deep understanding of customer journey and product lifecycle.
                    """)

            hudson, lawson = st.columns(2)
            hudson.info("Project Manager")
            lawson.info("Lead Software Tester")

            hudson_pics = Image.open("resources/imgs/hudson (2).png")
            lawson_pics = Image.open("resources/imgs/lawson (2).png")
            
            with hudson: #Bodine's profile and picture
                st.header("Hudson")
                st.image(hudson_pics)

                with st.expander("Brief Bio"):
                    st.write("""
                    Hudson is a Senor Machine Learning engineer with around 8 years of professional IT experience in Machine Learning
                    statistics modelling, Predictive modelling, Data Analytics, Data modelling, Data Architecture, Data Analysis, Data
                    mining, Text mining, Natural Language Processing(NLP), Artificial Intelligence algorithms, Business intelligence (BI),
                    analytics module (like Decision Trees, Linear and Logistics regression), Hadoop, R, Python, Spark, Scala, MS Excel and SQL.

                    He is proficient in managing the entire Data Science project lifecycle and actively involved in the phase of project
                    lifecycle including data acquisition, data cleaning, features engineering and statistical modelling.

                    """)


            with lawson: #Seyi's profile and picture
                st.header("Name")
                st.image(lawson_pics)

                with st.expander("Brief Bio"):
                    st.write("""
                    Lawson is an accomplished Quality Assurance tester with over 3 years experience in Software Testing and Quality Assurance.
                    He has a solid understanding in Software Development Life Cycle, Software Testing Lifecycle, bug lifecycle and testing
                    diiferent procedure.
                    """)
            

        
            
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
