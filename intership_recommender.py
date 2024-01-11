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
from bs4 import BeautifulSoup
from io import BytesIO

# POS TAG AND Word Lemmatizer

#{ Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
#}
POS_LIST = [NOUN, VERB, ADJ, ADV]

NUM_POSTING = 50

TOP_N_KEYWORDS = 5

def main():
    """
    Main function for the web app.

    Downloads necessary NLTK resources, sets up CSS styling, and creates the file uploader and user interface.

    Returns:
    None
    """

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Add CSS Style 
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    # Set the title of your app
    st.title('Internship Recommendations Based On Resume')
    st.markdown("**Author: Thang Truong**")
    st.markdown("[GitHub Repository](https://github.com/BrianTruong23/job_recommendation)")
    # Write a short description
    st.markdown("""
    This web app matches your resume with available job postings to help you find relevant job opportunities.
    Upload your resume and let the magic happen!             
    - This is a prototype to demonstrate the power of **natural language processing especially TF-IDF and cosine similarity.**
    - It can also display top keywords extracted from the resume so that the user will know what keywords are important in their resume.
    """)

    # Create a file uploader component
    uploaded_file = st.file_uploader("Upload a PDF, TXT or DOCX file", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size} bytes")
        resume = read_pdf(uploaded_file)
        resume = pre_process_resume(resume)
        job_df = get_job_df()
        df_resume_sorted = post_process_table(resume, job_df)
        display_features_slider(resume, job_df)
         # Create a button for exporting to CSV
        download_csv(df_resume_sorted)
        st.write(df_resume_sorted.to_html(escape=False), unsafe_allow_html=True)


def download_csv(df):
    st.markdown("**Download the internship postings as CSV File**")
    filename_csv = "internship_posting.csv"
    filename_excel = "internship_posting.xlsx"
    # Create a BytesIO buffer for storing the CSV data
    csv_buffer = BytesIO()
    excel_buffer = BytesIO()

    # Export the DataFrame to CSV and save it to the BytesIO buffer
    df.to_csv(csv_buffer, index=False)
    # df.to_excel(excel_buffer, index=False)

    # Create download buttons for CSV and Excel files
    csv_button = st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=filename_csv,
        mime="text/csv",
    )
    excel_button = st.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name=filename_csv,
        mime="text/xlsx",
    )

    if csv_button or excel_button:
        st.success(f"Data exported to {filename_csv if csv_button else filename_excel}")


def display_features_slider(resume, job_df):

    selected_features= st.slider(f"Select keywords extracted from your resume:", min_value=1, max_value=100, value=5)
    top_features = get_top_features(resume, job_df)
    st.markdown(f"**Selected {selected_features} keywords:**")
    data = pd.DataFrame({"Top Keywords": top_features[:selected_features]})
    # Display the DataFrame with a maximum of 10 rows
    st.table(data)

def get_top_features(resume_text, job_df):

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['data'])
    
    # Calculate the TF-IDF vector for the input word
    resume_text_vector = tfidf_vectorizer.transform([resume_text])

    # Get feature names from the TF-IDF vectorizer
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # Get the indices of features sorted by TF-IDF scores
    feature_indices = np.argsort(resume_text_vector.toarray()[0])[::-1]

    # Extract the top N features (keywords and skills)
    top_features = feature_names[feature_indices]

    return top_features

def get_job_df():
    """
    Pre-processes the job database

    Returns:
    job_df: DataFrame after pre-processing
    """
     
    job_data_list = return_data_list()
    job_df = pd.DataFrame(job_data_list, columns=['Company', 'Role', 'Location', 'Application/Link', 'Date Posted'])
    job_df = pre_process_data_job(job_df)
    return job_df

def post_process_table(uploaded_file, job_df):
    """
    Post-processes the job recommendations table after uploading a resume.

    Parameters:
    uploaded_file (str): The uploaded resume file.

    Returns:
    str: HTML-formatted table of job recommendations.
    """

    df_resume = return_table_job(str(uploaded_file), job_df)
    df_resume_sorted = df_resume.head(NUM_POSTING).sort_index(ascending=True)
    df_resume_sorted['Application/Link'] = df_resume_sorted['Application/Link'].apply(make_clickable)


    df_resume_sorted = df_resume_sorted.reset_index(drop=True).iloc[:, 0:-1]
    df_resume_sorted.index = df_resume_sorted.index + 1
    return df_resume_sorted



def make_clickable(link):
    """
    Create a clickable HTML link.

    Parameters:
    str: link - The URL for the hyperlink.

    Returns:
    str: An HTML-formatted string containing a clickable link.
    """
    text = "Apply Link"
    return f'<a href="{link}">{text}</a>'


def read_pdf(pdf_file):
    """
    Read text content from a PDF file.

    Parameters:
    str: pdf_file - The path to the PDF file.

    Returns:
    str: Text content extracted from the PDF file.
    """
    # Get pdf file 
    # Return text of the resume
    reader = PdfReader(pdf_file)
    # number_of_pages = len(reader.pages)
    page = reader.pages[0]
    resume = page.extract_text()
    return resume


def return_data_list():
    """
    Extract job data from an HTML file containing a table.

    Reads the content of the HTML file, parses it with BeautifulSoup,
    and extracts job data from the table.

    Returns:
    list: A list of lists containing job data, where each inner list
          represents a row in the table with job information.
    """
    job_data_list = []
    
    # Read the content of the HTML file
    with open('WebCrawler/table.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table element (adjust 'table' if you have a different tag)
    table = soup.find('table')

    # Find all <tr> elements within the <tbody>
    tr_elements = table.find_all('tr')

    # Iterate through the <tr> elements and extract the second <td>
    for i in range(len(tr_elements)):
        tr = tr_elements[i]
        td_ele = tr.find_all('td')
        td_list = [td.text.strip() if td.text.strip() else td.find('a').get('href') for td in td_ele]
        job_data_list.append(td_list)
    
    return job_data_list


def keep_alpha_char(text):
    """
    Remove non-alphabetic characters from the input text, keeping only alphabetical characters.

    Parameters:
    - text (str): The input text containing alphanumeric and non-alphanumeric characters.

    Returns:
    str: A cleaned string containing only alphabetical characters.
    """
    alpha_only_string = re.sub(r'[^a-zA-Z]', ' ', text)
    cleaned_string = re.sub(r'\s+', ' ', alpha_only_string)
    return cleaned_string



def nltk_pos_tagger(nltk_tag):
    """
    Map NLTK POS tags to WordNet POS tags.

    Parameters:
    - nltk_tag (str): The POS tag assigned by NLTK.

    Returns:
    int or None: WordNet POS tag corresponding to the input NLTK POS tag, or None if not recognized.
    """
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
    """
    Lemmatize the words in the input sentence.

    Parameters:
    - sentence (str): The input sentence to be lemmatized.

    Returns:
    str: The lemmatized sentence.
    """
    lemmatizer = WordNetLemmatizer()
    
    # Part-of-speech tagging using NLTK
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    
    # Map NLTK POS tags to WordNet POS tags
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    
    lemmatized_sentence = []
    
    # Lemmatize each word based on its POS tag
    for word, tag in wordnet_tagged:
        if tag is not None:      
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    
    # Join the lemmatized words back into a sentence
    return " ".join(lemmatized_sentence)


def remove_stop_words(text):
    """
    Remove English stop words from the input text.

    Parameters:
    - text (str): The input text from which stop words will be removed.

    Returns:
    str: The input text with English stop words removed.
    """
    # Tokenize the text
    words = nltk.word_tokenize(str(text))

    # Get the list of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the list of words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a text
    filtered_text = ' '.join(filtered_words)

    return filtered_text



def recommend_job(resume_text, tfidf_matrix, tfidf_vectorizer, df):
    """
    Recommends jobs based on an input word using TF-IDF and cosine similarity.

    Parameters:
    - resume_text (str): The input word or text for which job recommendations are sought.
    - tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix representing job descriptions.
    - tfidf_vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for vectorizing input words.
    - df (pd.DataFrame): The DataFrame containing job information.

    Returns:
    pd.DataFrame: A table of recommended jobs sorted by similarity to the input word.
    """
    # Calculate the TF-IDF vector for the input word
    resume_text_vector = tfidf_vectorizer.transform([resume_text])

    # Calculate cosine similarities between the input word vector and job vectors
    cosine_similarities = cosine_similarity(resume_text_vector, tfidf_matrix)

    # Get indices of jobs sorted by similarity (highest to lowest)
    job_indices = cosine_similarities.argsort()[0][::-1]

    # Extract the jobs corresponding to the top recommendations
    top_recommendations_full = [df.iloc[index] for index in job_indices]



    return pd.DataFrame(top_recommendations_full)

def pre_process_resume(resume_text):
    """
    Preprocesses a resume text by removing stop words, lemmatizing, and keeping only alphabet characters.

    Parameters:
    - resume_text (str): The text content of the resume.

    Returns:
    str: The preprocessed resume text.
    """
    # Remove non-alphabetic characters and keep only alphabet characters
    resume_text = keep_alpha_char(resume_text)

    # Lemmatize the words in the resume text
    resume_text = lemmatize_sentence(resume_text)

    # Remove stop words from the resume text
    resume_text = remove_stop_words(resume_text)

    # Convert the resume text to lowercase
    resume_text = resume_text.lower()

    return resume_text


def pre_process_data_job(job_df):
    """
    Preprocess the job_df database by removing stop words, returning the words to their base form,
    and keeping only alphabet characters.

    Parameters:
    - job_df (pd.DataFrame): The job database containing job descriptions.
      job_df["Role"] is the column that contains the title and the role of the internship.

    Returns:
    pd.DataFrame: A table of recommended jobs that has the pre-processed data for the column "Role".
    """
    # Drop rows with missing values in the "Role" column
    job_df.dropna(subset=['Role'], inplace=True)

    # Apply preprocessing steps directly to the original DataFrame
    job_df['data'] = job_df['Role'].apply(keep_alpha_char)
    job_df['data'] = job_df['data'].apply(lemmatize_sentence)
    job_df['data'] = job_df['data'].apply(remove_stop_words)
    job_df['data'] = job_df['data'].str.lower()

    # Removing string that has 🔒 (which means no longer active) in application/link
    job_df["Application/Link"].replace('🔒', pd.NA, inplace=True)
    job_df.dropna(inplace = True)

    return job_df


def return_table_job(resume_text, job_df):
    """
    Generates a table of recommended jobs based on the provided resume text and a job database.

    Parameters:
    - text (str): The resume text or content of the uploaded resume file.
    - job_df (pd.DataFrame): The job database containing job descriptions.

    Returns:
    pd.DataFrame: A table of recommended jobs sorted by relevance to the provided resume.
    The table includes columns for job titles, locations, dates, and links to job postings.
    """

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['data'])
    
    # Recommend jobs using cosine similarity
    recommended_jobs = recommend_job(resume_text, tfidf_matrix, tfidf_vectorizer, job_df)
    
    # Drop the first column, which is the similarity score
    recommended_jobs = recommended_jobs.drop(columns=recommended_jobs.columns[0])

    return recommended_jobs


if __name__ == "__main__":
    main()
