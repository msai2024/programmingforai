import re

from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL
import csv
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
app.config['MYSQL_PASSWORD'] = '$12Ayodeji34'
app.config['MYSQL_DB'] = 'sys'
app.secret_key = os.urandom(24)

mysql = MySQL(app)

# Load spaCy and Sentence Transformer models
# nlp = spacy.load("./en_Resume_Matching_Keywords-any-py3-none-any.whl")
nlp = spacy.load("en_Resume_Matching_Keywords")
model_path = "./Matching-job-descriptions-and-resumes/msmarco-distilbert-base-tas-b-final"
model = SentenceTransformer(model_path)


def create_unprocessed_jobs_data_table():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS unprocessed_jobs_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Experience TEXT, Qualifications TEXT, Salary_Range TEXT, Country TEXT, Work_Type TEXT,
        Job_Posting_Date TEXT, Preference TEXT, Contact_Person TEXT, Job_Title TEXT,
        Role TEXT, Job_Portal TEXT, Job_Description TEXT, Benefits TEXT,
        skills TEXT, Responsibilities TEXT, Company TEXT, Company_Profile TEXT 
    )
    """

    cursor = mysql.connection.cursor()
    cursor.execute(create_table_query)
    mysql.connection.commit()
    cursor.close()


def create_cleaned_jobs_data_table():
    create_table_query = """
        CREATE TABLE IF NOT EXISTS cleaned_jobs_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Minimum_Salary_Range TEXT, Maximum_Salary_Range TEXT, Minimum_Experience TEXT, Maximum_Experience TEXT,
            Qualifications TEXT, Country TEXT, Work_Type TEXT, Job_Posting_Date TEXT, Preference TEXT, 
            Role TEXT, Job_Description TEXT, skills TEXT, Company TEXT
        )
        """

    cursor = mysql.connection.cursor()
    cursor.execute(create_table_query)
    mysql.connection.commit()
    cursor.close()


def write_unprocessed_jobs_into_database():
    dict_list = list()
    with (open('./CSV/job_listings.csv', 'r') as csvfile):
        csvreader = csv.reader(csvfile)
        next(csvreader)
        cursor = mysql.connection.cursor()
        cursor.execute("SHOW TABLES LIKE 'unprocessed_jobs_data'")
        table_exists = cursor.fetchone()
        if not table_exists:
            create_unprocessed_jobs_data_table()
        else:
            cursor.execute("delete from unprocessed_jobs_data")  # clean the database
        for i, row in enumerate(csvreader):
            if i >= 30:
                break
            dict_list.append(
                {'Experience': row[0], 'Qualifications': row[1], 'Salary_Range': row[2], 'Country': row[3],
                 'Work_Type': row[4], 'Job_Posting_Date': row[5], 'Preference': row[6], 'Contact_Person': row[7],
                 'Job_Title': row[8], 'Role': row[9], 'Job_Portal': row[10], 'Job_Description': row[11],
                 'Benefits': row[12], 'skills': row[13], 'Responsibilities': row[14], 'Company': row[15],
                 'Company_Profile': row[16]})
        for item in dict_list:
            sql = (
                "INSERT INTO unprocessed_jobs_data(Experience, Qualifications, Salary_Range, Country, Work_Type, Job_Posting_Date, Preference, Contact_Person, "
                "Job_Title, Role, Job_Portal, Job_Description, Benefits, skills, Responsibilities, Company, Company_Profile) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
            val1 = item['Experience'], item['Qualifications'], item['Salary_Range'], item['Country'], item['Work_Type'], \
                item['Job_Posting_Date'], item['Preference'], item['Contact_Person'], item['Job_Title']
            val2 = item['Role'], item['Job_Portal'], item['Job_Description'], item['Benefits'], item['skills'], item[
                'Responsibilities'], item['Company'], item['Company_Profile']
            cursor.execute(sql, val1 + val2)
        mysql.connection.commit()
        cursor.close()


def write_clean_data_into_database():
    query = "SELECT Experience, Qualifications, Salary_Range, Country, Work_Type, Job_Posting_Date, Preference, Role, Job_Description, skills, Company FROM unprocessed_jobs_data"
    cursor = mysql.connection.cursor()
    cursor.execute(query)
    unprocessed_jobs = cursor.fetchall()
    cursor.close()
    unprocessed_jobs_df = pd.DataFrame(unprocessed_jobs,
                                       columns=["Experience", "Qualifications", "Salary_Range", "Country", "Work_Type",
                                                "Job_Posting_Date", "Preference", "Role", "Job_Description", "skills",
                                                "Company"])
    unprocessed_jobs_df["Minimum_Experience"] = unprocessed_jobs_df["Experience"].apply(
        lambda x: re.split(r' to | Years', x)[0])
    unprocessed_jobs_df["Maximum_Experience"] = unprocessed_jobs_df["Experience"].apply(
        lambda x: re.split(r' to | Years', x)[1])
    unprocessed_jobs_df["Minimum_Salary_Range"] = unprocessed_jobs_df["Salary_Range"].apply(
        lambda x: re.split(r'-', x)[0]).apply(lambda x: x.replace('K', "000").replace('$', ""))
    unprocessed_jobs_df["Maximum_Salary_Range"] = unprocessed_jobs_df["Salary_Range"].apply(
        lambda x: re.split(r'-', x)[1]).apply(lambda x: x.replace('K', "000").replace('$', ""))
    unprocessed_jobs_df.drop(columns=["Experience", "Salary_Range"]).to_csv('./CSV/processed_jobs_data.csv',
                                                                            index=False)

    dict_list = list()
    with (open('./CSV/processed_jobs_data.csv', 'r') as csvfile):
        csvreader = csv.reader(csvfile)
        next(csvreader)
        cursor = mysql.connection.cursor()
        cursor.execute("SHOW TABLES LIKE 'cleaned_jobs_data'")
        table_exists = cursor.fetchone()
        if not table_exists:
            create_cleaned_jobs_data_table()
        else:
            cursor.execute("delete from cleaned_jobs_data")  # clean the database
        for i, row in enumerate(csvreader):
            dict_list.append(
                {'Qualifications': row[0], 'Country': row[1], 'Work_Type': row[2], 'Job_Posting_Date': row[3],
                 'Preference': row[4], 'Role': row[5], 'Job_Description': row[6], 'skills': row[7], 'Company': row[8],
                 'Minimum_Experience': row[9], 'Maximum_Experience': row[10], 'Minimum_Salary_Range': row[11],
                 'Maximum_Salary_Range': row[12]})
        for item in dict_list:
            sql = "INSERT INTO cleaned_jobs_data(Qualifications, Country, Work_Type, Job_Posting_Date, Preference, Role, Job_Description, skills, Company, Minimum_Experience, Maximum_Experience, Minimum_Salary_Range, Maximum_Salary_Range ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val1 = item['Qualifications'], item['Country'], item['Work_Type'], item['Job_Posting_Date'], item[
                'Preference'], item['Role'], item['Job_Description'], item['skills'], item['Company'], item[
                'Minimum_Experience'], item['Maximum_Experience'], item['Minimum_Salary_Range'], item[
                'Maximum_Salary_Range']
            cursor.execute(sql, val1)
        mysql.connection.commit()
        cursor.close()


with app.app_context():
    write_unprocessed_jobs_into_database()
    write_clean_data_into_database()


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
        print(
            f"Job Title: {job_title}, Skills: {job_skills}, Company: {company}, Salary: {salary_range}, Role: {role}")  # Print job details
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
