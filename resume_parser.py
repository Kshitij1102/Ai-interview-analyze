import spacy
from pdfminer.high_level import extract_text
import whisper
import os
import json
import streamlit as st
from textblob import TextBlob

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
        print(f"Error extracting text: {e}")
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
        print(f"Error transcribing audio: {e}")
        return ""

def analyze_sentiment(text):
    """Analyzes sentiment of the transcribed text."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "Neutral"
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    return sentiment, polarity

def save_results_to_file(filename, data):
    """Save extracted data to a JSON file."""
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    st.title("AI Interview Analyzer")
    
    uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    uploaded_audio = st.file_uploader("Upload your Interview Audio (MP3)", type=["mp3"])
    
    results = {}
    
    if uploaded_resume is not None:
        resume_text = extract_resume_text(uploaded_resume)
        keywords = extract_keywords(resume_text)
        st.write("Extracted Keywords (Skills & Specializations):", keywords)
        results["resume_keywords"] = keywords
    
    if uploaded_audio is not None:
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_audio.read())
        transcript = transcribe_audio("temp_audio.mp3")
        sentiment, score = analyze_sentiment(transcript)
        st.write("\nInterview Transcript:", transcript)
        st.write("\nSentiment Analysis: ", sentiment, "(Score:", score, ")")
        results["interview_transcript"] = transcript
        results["sentiment"] = sentiment
        results["sentiment_score"] = score
    
    if results:
        save_results_to_file("interview_analysis_results.json", results)
