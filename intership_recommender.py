import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# POS TAG AND Word Lemmatizer

#{ Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
#}
POS_LIST = [NOUN, VERB, ADJ, ADV]


NUM_POSTING = 50

def main():

    # Add CSS Style 
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
    # Set the title of your app
    st.title('Job Posting Based On Resume')
    # Create a file uploader component
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size} bytes")
        df_resume_sorted = post_process_table(uploaded_file)
        st.write(df_resume_sorted, unsafe_allow_html=True)


def post_process_table(uploaded_file):
    job_data_list = return_data_list()
    job_df = pd.DataFrame(job_data_list, columns = ['Company', 'Role', 'Location', 'Application/Link', 'Date Posted'])
    job_df = pre_process_data(job_df)

    df_resume = return_table_job(str(uploaded_file), job_df)
    df_resume_sorted = df_resume.head(NUM_POSTING).sort_index(ascending = True)
    # df_resume_sorted['Application/Link'] = df_resume_sorted['Application/Link'].apply(lambda x: f'<a href = {x} target = "_blank">Apply </a>' )

    df_resume_sorted['Application/Link'] = df_resume_sorted['Application/Link'].apply(make_clickable)
    df_resume_sorted = df_resume_sorted.reset_index(drop = True).iloc[:, 0:-1]
    df_resume_sorted.index = df_resume_sorted.index  + 1
    df_resume_sorted = df_resume_sorted.to_html(escape=False)
    return df_resume_sorted


def make_clickable(link):
    text = "Apply Link"
    return f'<a href="{link}">{text}</a>'

def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    resume = page.extract_text()
    return resume

def return_data_list():
    import requests
    from bs4 import BeautifulSoup

    # Define the URL of the webpage you want to scrape
    url = "https://github.com/SimplifyJobs/Summer2024-Internships"

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    job_data_list = []

    # Check if the request was not successful (status code 200)
    if response.status_code != 200:
        raise ValueError("unable to retrieve the webpage")

    # Parse the HTML content of the page using Beautiful Soup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the <tbody> element
    tbody = soup.find('tbody')

    # Find all <tr> elements within the <tbody>
    
    tr_elements = tbody.find_all('tr')

    # Iterate through the <tr> elements and extract the second <td>
    for i in range(len(tr_elements)):
        tr = tr_elements[i]
        td_ele = tr.find_all('td')
        td_list = [td.text.strip() if td.text.strip() else td.find('a').get('href') for td in td_ele]
        job_data_list.append(td_list)    
   
    
    return job_data_list

def keep_alpha_char(text):
    alpha_only_string = re.sub(r'[^a-zA-Z]', ' ', text)
    cleaned_string = re.sub(r'\s+', ' ', alpha_only_string)
    return cleaned_string


def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

    
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []
    
    for word, tag in wordnet_tagged:
        if tag is not None:      
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def remove_stop_words(text):
    # tokenize the text 
    words = nltk.word_tokenize(str(text))

    # Get the list of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the list of words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    filtered_text = ' '.join(filtered_words)
    
    return filtered_text


# Building the Recommendation Engine
def recommend_job(input_word, tfidf_matrix, tfidf_vectorizer, df):
    # Calculate the TF-IDF vector for the input word
    input_word_vector = tfidf_vectorizer.transform([input_word])

    # Calculate cosine similarities between the input word vector and job vectors
    cosine_similarities = cosine_similarity(input_word_vector, tfidf_matrix)

    # Get indices of jobs sorted by similarity (highest to lowest)
    job_indices = cosine_similarities.argsort()[0][::-1]

    # Extract the job titles corresponding to the top recommendations
    
    top_recommendations_full = []
    for i in range(len(job_indices)):
        job_full = df.iloc[job_indices[i]]
        top_recommendations_full.append(job_full)
        
    return pd.DataFrame(top_recommendations_full)


def pre_process_data(job_df):
    job_df['data'] = job_df['Role'].apply(keep_alpha_char)
    job_df['data']= job_df['data'].apply(lemmatize_sentence)
    job_df['data'] = job_df['data'].apply(remove_stop_words)
    job_df['data'] = job_df['data'].str.lower()
    return job_df


def return_table_job(text, job_df):
    # More Text processing and TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['data'])
    recommended_jobs = recommend_job(text, tfidf_matrix, tfidf_vectorizer, job_df)
    recommended_jobs = recommended_jobs.drop(columns=recommended_jobs.columns[0])
    return recommended_jobs

if __name__ == "__main__":
    main()
