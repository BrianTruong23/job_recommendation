# Internship Recommendation System using Machine Learning

Accessed Website: https://intershiprecommendation.streamlit.app/

# Why this project is created? 
Internship searching is a challenging task, and the process of finding the right opportunities that align with your skills and aspirations can be time-consuming. Therefore, the goal of this system is to simplify and enhance the internship search experience. By uploading your resume, the system analyzes its key strengths and matches them with a curated list of internship postings. This personalized approach aims to help users discover relevant opportunities more efficiently.

### The primary objectives of this project include:

#### Facilitating Easier Internship Discovery: 
- The system leverages natural language processing techniques, such as TF-IDF and cosine similarity, to recommend internship postings that closely match the content of your resume. This assists users in discovering opportunities that align with their qualifications and preferences.

#### Empowering Users to Identify Resume Strengths: 
- In addition to internship recommendations, the system provides insights into the key strengths of your resume. By understanding which skills and experiences are highlighted, users can gain valuable insights into how their profiles are perceived by potential employers.

#### Learning and Skill Development: 
- Behind the scenes, this project serves as a learning experience in the field of data science application development for me. It involves the implementation of various natural language processing techniques and optimization strategies to enhance the performance and efficiency of the recommendation system.

# What is the process look like for the users?
#### Step 1: 
- Upload the Resume as PDF 
#### Step 2: 
- Get your internship based on your resume
#### It is that easy and effective to search for your internship postings based on your resume. 

# Data Processing:
I scraped the web using Web Driver Selenium and Beautiful Soup to extract 1898 job postings. After getting the job postings, I need to elminate all of the job postings that are inactive (empty string in the Application/Link) to get 657 job postings left. After that, all the job title and roles are processed by eliminating all of the stopwords, returning them to their roots (lemmatizing them), keep only alphabet characters and lowering-case them. 

# Machine Learning Process:
My machine learning techniques implement the use of TF-IDF (Term Frequency - Inverse Document Frequency) vectorization technique and cosine similarity to match the descriptions of the job postings to the key words identified by the vectorization methods of the resume. The vectors effectively identifies the significance of the each word of the resume and job posting, and cosine similarity matches job postings with the resume keywords. 

# App Production Visualization

<img src="https://github.com/BrianTruong23/job_recommendation/assets/40693511/a7e38fa1-e4c6-4288-a2e5-fd6dfb18ed17" alt="app_production" width="800" height="800">

# Demo 
<img width="736" alt="demo" src="https://github.com/BrianTruong23/job_recommendation/assets/40693511/d2b04098-66c6-4859-8771-625da8549385">

# Testing 
After examing the dataset, a lot of jobs are in software engineering industry and some are in finance industry (with a majority in software engineering field). Therefore, I proposed two documents containg keywords relating two subfields in software engineering for a fair and complete testing process: data science and web developer. Then, the system will try to recommend corresponding job postings based on the keywords. After that, in order to evaluate the accuracy, I can calculate the percentage of accuracy by counting all of the roles relating to those two fields respectively out of the total job postings. 

**Overall Statistics:**
Total Roles: 657
Data Science Roles: 71
Web Development Roles: 45

__Results of Testing for Data Science Roles:__
For the CSV file named data_science_roles_testing.csv:
Total Roles: 40
Data Science Roles: 28
__Percentage Correct: 70.0%__

__Results of Testing for Web Development Roles:__
For the CSV file named web_dev_roles_testing.csv:
Total Roles: 40
Web Development Roles: 32
__Percentage Correct: 80.0%__





