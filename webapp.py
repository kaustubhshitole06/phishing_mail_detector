
import os
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

email=pd.read_csv('phishing_email.csv')

print(email.head())

print(email.tail())

print(email.describe())

print(email['label'].value_counts())

print(email.shape)

email.drop_duplicates(inplace=True)
email['label']=email['label'].replace([1,0],['Phishing','Legitimate'])

print(email.shape)

text=email['text_combined']

label=email['label']
email['label'].value_counts()

(text_train, text_test, label_train, label_test )= train_test_split(text, label, test_size=0.2, random_state=42)

print(text_train.head())

cv=CountVectorizer(stop_words='english')
features=cv.fit_transform(text_train)

model=MultinomialNB()
model.fit(features,label_train)

print(text_test.head())

fet_test=cv.transform(text_test)
print(model.score(fet_test,label_test))

def predict(message):
    message=cv.transform([message]).toarray()
    result=model.predict(message)
    return result

    
import streamlit as st

########################################################################
import time
import random

st.set_page_config(page_title="Phishing Email Detector", page_icon="🕵️‍♂️", layout="centered")

st.markdown("<h1 style='text-align: center; color: red;'>🕵️‍♂️ Phishing Email Detector 📧</h1>", unsafe_allow_html=True)

st.info("This tool helps you detect phishing emails. Enter text to analyze.")

email_text = st.text_area("Paste the email content here:", height=150)


#def predict_email(text):
    #time.sleep(1)  
    #return "Phishing" if random.random() > 0.5 else "Legitimate"

# Button to Analyze Email
if st.button("Analyze Email"):
            if len(email_text) < 1:
                st.warning("⚠️ Please enter an email to analyze.")
            else:
                result2 = predict(email_text)
                st.success(f"Result: {result2}")



st.markdown("<marquee behavior=alternate bgcolor='red' scrollamount=10 height=42><h3>Stay Safe! 🛡️</h3></marquee>",unsafe_allow_html=True)

