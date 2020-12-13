import streamlit as st
import joblib,os
import sqlite3
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer



# title_vectorizer = open("final_news_cv_vectorizer.pkl","rb")
# title_cv = joblib.load(title_vectorizer)

# FUNCTIONS

def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def getDTMByTFIDF(titles,nfeatures):
    
    tfIdf_vectorizer = TfidfVectorizer(max_features=nfeatures)
    dtm = tfIdf_vectorizer.fit_transform(titles).toarray()
    return dtm, tfIdf_vectorizer


def main():
    st.title("Journal Paper Classifier")
    # st.subheader("ML App with Streamlit")

    paper_title = st.text_area("Add your title here" ,"Type Here")
    paper_title =[paper_title]

    models =  ['SVM', 'NB', 'RF']
    model_choice = st.selectbox("Select Model",models)

    if st.button("Classify"):
        st.text("Original Text::\n{}".format(paper_title))
        # vect_text = title_cv.transform([paper_title]).toarray()
# TO DO: the vectorization is giving errors
        if model_choice == 'SVM':
            predictor = load_prediction_models("SVM_model.sav")
            vect = load_prediction_models('vect_model.sav')
            tf1_new = TfidfVectorizer(max_features = None, vocabulary = vect.vocabulary_)
            vect_text = tf1_new.fit_transform(paper_title)

            # vect_text = vect.transform(paper_title)
            prediction = predictor.predict(vect_text)
            st.write(prediction)
            st.success("Your predicted Journal is: {}".format(prediction[0]))

        elif model_choice == 'NB':
            predictor = load_prediction_models("NB_model.sav")
            vect = load_prediction_models('vect_model.sav')
            tf1_new = TfidfVectorizer(max_features = None, vocabulary = vect.vocabulary_)
            vect_text = tf1_new.fit_transform(paper_title)

            # vect_text = vect.transform(paper_title)
            prediction = predictor.predict(vect_text)
            st.write(prediction)
            st.success("Your predicted Journal is: {}".format(prediction[0]))

        elif model_choice == 'RF':
            predictor = load_prediction_models("rf_model.sav")
            vect = load_prediction_models('vect_model.sav')
            tf1_new = TfidfVectorizer(max_features = None, vocabulary = vect.vocabulary_)
            vect_text = tf1_new.fit_transform(paper_title)

            # vect_text = vect.transform(paper_title)
            prediction = predictor.predict(vect_text)
            st.write(prediction)

            # final_result = get_key(prediction,prediction_labels)
            st.success("Your predicted Journal is: {}".format(prediction[0]))
            

    st.sidebar.title("Journal Paper Classifier")
    st.sidebar.markdown('How does it work?')
    st.sidebar.markdown('Finding and selecting a suitable academic journal is always challenging especially for young researchers')
    st.sidebar.markdown('Various classification algorithms will be used to research papers by journal along with feature selection and dimensionality reduction methods using scikit-learn')
    # annotated_text(
    # "The development of a ",
    # ("Bioreactor", "aligned word", "#8ef"),
    # " for ",
    # ("biocatalyst", "aligned word", "#faa"),
    # ("production"))



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

