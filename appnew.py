from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import spacy
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text
import pandas as pd
import numpy as np
import os
from io import BytesIO
from tempfile import NamedTemporaryFile

# Flask app configuration
app = Flask(__name__, static_folder='static')

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'sys'
app.secret_key = os.urandom(24)

mysql = MySQL(app)

# Load spaCy and Sentence Transformer models
nlp = spacy.load("en_Resume_Matching_Keywords")
model_path = "E:/MSc AI/Sem1 - Data Analytics/Project/Spacy_Transformer/model/Matching-job-descriptions-and-resumes/msmarco-distilbert-base-tas-b-final"
model = SentenceTransformer(model_path)

def fetch_jobs_from_database(page=1, per_page=1000, search=None):
    """Fetch job titles and skills from the database with pagination and optional search."""
    with app.app_context():
        cursor = mysql.connection.cursor()
        base_query = "SELECT Job_Title, Skills, Company, Salary_Range, Role FROM sys.job_listings"
        count_query = "SELECT COUNT(*) FROM sys.job_listings"

        if search:
            base_query += " WHERE Job_Title LIKE %s OR Skills LIKE %s"
            count_query += " WHERE Job_Title LIKE %s OR Skills LIKE %s"
            search_pattern = f'%{search}%'
            cursor.execute(count_query, [search_pattern, search_pattern])
        else:
            cursor.execute(count_query)

        total_count = cursor.fetchone()[0]
        total_pages = (total_count + per_page - 1) // per_page

        if search:
            cursor.execute(base_query + " LIMIT %s, %s", ((page - 1) * per_page, per_page))
        else:
            cursor.execute(base_query + " LIMIT %s, %s", ((page - 1) * per_page, per_page))
        
        jobs = cursor.fetchall()
        cursor.close()
        return jobs, total_pages

def extract_text_from_pdf(pdf_file):
    """ Extract text content from a PDF file. """
    try:
        pdf_binary = pdf_file.read()  # Get binary data of the PDF file
        return extract_text(BytesIO(pdf_binary))  # Pass the binary data to the extraction function
    except Exception as e:
        print(f"Error extracting text from {pdf_file.filename}: {e}")
        return None

def extract_skills(text):
    """ Extract skills from text using spaCy. """
    doc = nlp(text)
    return ' '.join([ent.text for ent in doc.ents if ent.label_ == 'SKILLS'])

def get_embeddings(text):
    """ Generate embeddings for given text using Sentence Transformers. """
    return model.encode(text)

def compute_similarity(embedding1, embedding2):
    """ Compute cosine similarity between two embeddings. """
    return util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()

def match_resume_to_jobs(resume_text, jobs):
    if not resume_text:
        return pd.DataFrame()  # Return empty DataFrame if text extraction fails

    resume_skills_text = extract_skills(resume_text)
    resume_skills_embedding = get_embeddings(resume_skills_text)
    print(f"Extracted skills from resume: {resume_skills_text}") 
    
    results = []
    for job_title, job_skills, company, salary_range, role in jobs:
        print(f"Job Title: {job_title}, Skills: {job_skills}, Company: {company}, Salary: {salary_range}, Role: {role}")  # Print job details
        job_skills_text = job_skills  # Assuming skills are already a concatenated string
        job_skills_embedding = get_embeddings(job_skills_text)
        similarity_score = compute_similarity(resume_skills_embedding, job_skills_embedding)
        results.append((job_title, company, salary_range, role, similarity_score))

    results_df = pd.DataFrame(results, columns=['Job Title', 'Company', 'Salary Range', 'Role', 'Similarity Score'])
    results_df.sort_values(by='Similarity Score', ascending=False, inplace=True)
    return results_df


# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/jobs')
def jobs():
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Consider defining this more globally or as a parameter
    search = request.args.get('search', '')
    jobs, total_pages = fetch_jobs_from_database(page, per_page, search)
    page = max(1, min(page, total_pages))  # Correct page number if out of range
    return render_template('jobs.html', jobs=jobs, search=search, page=page, total_pages=total_pages)

@app.route('/upload-resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return render_template('index.html', message="No file part")
    
    resume_file = request.files['resume']
    if resume_file.filename == '':
        return render_template('index.html', message="No selected file")
    
    resume_text = extract_text_from_pdf(resume_file)
    if not resume_text:
        return render_template('index.html', message="Error extracting text from resume")
    
    # Assuming you want to fetch all jobs for the matching, adjust these parameters as needed
    page = 1
    per_page = 1000  # Large number, assuming it covers all jobs, adjust based on your actual data size
    jobs, total_pages = fetch_jobs_from_database(page, per_page)
    
    matches_df = match_resume_to_jobs(resume_text, jobs)
    
    # Get top 10 matches
    top_10_matches = matches_df.head(10).to_dict('records')
    return render_template('matches.html', matches=top_10_matches)  # Redirect to matches page

if __name__ == '__main__':
    app.run(debug=True)
