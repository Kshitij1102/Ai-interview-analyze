import spacy
from pdfminer.high_level import extract_text
import whisper
import os
import json
import streamlit as st
from textblob import TextBlob
import random
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing. Please check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Predefined skill database for filtering
SKILLS_DB = {"Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning", "NLP",
             "Cybersecurity", "Data Science", "AI", "TensorFlow", "PyTorch", "Django", 
             "Flask", "React", "Node.js", "Cloud Computing", "DevOps", "Linux", "Networking",
             "Docker", "Kubernetes"}

def extract_resume_text(pdf_path):
    """Extract text from a PDF resume."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def extract_keywords(text):
    """Extracts relevant skills and specializations using spaCy NLP."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    extracted_skills = set()
    for token in doc:
        clean_token = token.text.strip()
        if clean_token in SKILLS_DB:  # Match only predefined skills
            extracted_skills.add(clean_token)
    
    return list(extracted_skills)

def transcribe_audio(audio_path):
    """Transcribes speech from an audio file using Whisper."""
    try:
        model = whisper.load_model("base")  # Load the Whisper model
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

def analyze_sentiment(text):
    """Analyzes sentiment using GPT-3.5."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change from gpt-4 to gpt-3.5-turbo
            messages=[{"role": "system", "content": "Analyze the sentiment of the following text."},
                      {"role": "user", "content": text}],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return ""

def generate_questions(skills):
    """Generate interview questions based on extracted skills using GPT-3.5."""
    try:
        prompt = f"Generate interview questions based on these skills: {', '.join(skills)}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an expert interviewer."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def evaluate_resume_and_fit(keywords, job_role):
    """Evaluates if the candidate's resume matches the given job role and assigns a score."""
    try:
        prompt = f"Evaluate the resume skills {keywords} for the job role {job_role} and give a score (0-100) along with fit assessment."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an AI hiring assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error evaluating resume: {e}")
        return ""

def save_results_to_file(filename, data):
    """Save extracted data to a JSON file."""
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    st.success(f"Results saved to {filename}")

# Streamlit App
st.title("AI Interview Analyzer")

job_role = st.text_input("Enter the Job Role You Are Applying For")
uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
uploaded_audio = st.file_uploader("Upload your Interview Audio (MP3)", type=["mp3"])

results = {}

if job_role and uploaded_resume is not None:
    resume_text = extract_resume_text(uploaded_resume)
    if resume_text:
        keywords = extract_keywords(resume_text)
        st.subheader("Extracted Skills & Specializations:")
        st.write(keywords)
        results["resume_keywords"] = keywords
        
        interview_questions = generate_questions(keywords)
        st.subheader("Generated Interview Questions:")
        st.write(interview_questions)
        results["generated_questions"] = interview_questions
        
        evaluation = evaluate_resume_and_fit(keywords, job_role)
        st.subheader("Resume Evaluation Score and Fit:")
        st.write(evaluation)
        results["resume_evaluation"] = evaluation

if uploaded_audio is not None:
    audio_path = "temp_audio.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    transcript = transcribe_audio(audio_path)
    if transcript:
        sentiment_analysis = analyze_sentiment(transcript)
        st.subheader("Interview Transcript:")
        st.write(transcript)
        st.subheader("Sentiment Analysis:")
        st.write(f"Sentiment: {sentiment_analysis}")
        results["interview_transcript"] = transcript
        results["sentiment_analysis"] = sentiment_analysis

if results:
    save_results_to_file("interview_analysis_results.json", results)
