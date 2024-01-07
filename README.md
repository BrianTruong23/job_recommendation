# Internship Recommendation System using Machine Learning

# Why this project is created? 
Internship searching is hard! Thereofore, the goal of this system is to propose a lot of internship tailored to the upload of your resume.

## Step 1: Upload the Resume as PDF 
## Step 2: Get your job based on your resume 
It is that easy and effective to search for your job postings based on your resume. The jobs are updated as soon as the jobs are posted, and if you use Simplify to apply for jobs. It is even easier for you to apply for them.

# Database
The internship database is from Simplify: https://simplify.jobs/l/Top-Summer-Internships. 

# Machine Learning Process:
My machine learning techniques implement the use of TF-IDF (Term Frequency - Inverse Document Frequency) vectorization technique and cosine similarity to match the descriptions of the job postings to the key words identified by the vectorization methods of the resume. The vectors effectively identifies the significance of the each word of the resume and job posting, and cosine similarity matches job postings with the resume keywords. 

In Summary,
• Scraped the web for over 1000 job opportunities to collect 5 different columns: descriptions, titles, date, location and link
• Handled data cleaning to effectively remove stop words, lemmatize and vectorize sentences.
• Used 2 popular NLP technique, TF-IDF and cosine similarity, to match resume with job descriptions

<img width="860" alt="DemoJobSearch" src="https://github.com/BrianTruong23/job_recommendation/assets/40693511/d6860f70-cf0f-45a9-932d-131c7f65742a">
