import streamlit as st
import joblib,os
import sqlite3
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


title_vectorizer = open("final_news_cv_vectorizer.pkl","rb")
title_cv = joblib.load(title_vectorizer)

# FUNCTIONS

def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key


def main():
    st.title("Journal Paper Classifier")
    # st.subheader("ML App with Streamlit")

    paper_title = st.text_area("Add your title here" ,"Type Here")

    models =  ['SVM', 'NB', 'RF']
    model_choice = st.selectbox("Select Model",models)

    if st.button("Classify"):
        st.text("Original Text::\n{}".format(paper_title))
        vect_text = title_cv.transform([paper_title]).toarray()

        if model_choice == 'SVM':
            predictor = load_prediction_models("SVM_model.sav")
            prediction = predictor.predict(vect_text)

            final_result = get_key(prediction,prediction_labels)
            st.success("News Categorized as:: {}".format(final_result))
            

    st.sidebar.title("Journal Paper Classifier")



    # if st.checkbox("WordCloud"):
    # 		c_text = raw_text
    # 		wordcloud = WordCloud().generate(c_text)
    # 		plt.imshow(wordcloud,interpolation='bilinear')
    # 		plt.axis("off")
    # 		st.pyplot()







hide_streamlit_style = """
            <style>

            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__=='__main__':
    main()

