from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")

embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Prompt for generating questions
prompt_template = """
You are creating a technical interview. Based on the candidate's resume and job description, generate five distinct technical interview questions for a candidate, each with relevant follow-up questions.
Each primary question should focus on the candidate's core technical competencies based on their resume.

Candidate Resume Text: {candidate_resume}
Job Description: {job_description}

Return only the JSON-like format:
[
    {{
        "main_question": "<Primary question 1>",
        "follow": ["<Follow-up question 1.1>", "<Follow-up question 1.2>", ...]
    }},
    {{
        "main_question": "<Primary question 2>",
        "follow": ["<Follow-up question 2.1>", "<Follow-up question 2.2>", ...]
    }},
    ...
]
"""

# Initialize the LangChain prompt and chain for question generation
question_prompt = PromptTemplate(template=prompt_template, input_variables=["candidate_resume", "job_description"])
question_chain = question_prompt | llm

# Load PDF resumes, split text, and create JSON structure for output
def generate_interview_json(pdf_paths, job_description):
    interview_json = {}

    for pdf_path in pdf_paths:
        # Extract candidate name from filename or define a custom rule for naming
        candidate_name = pdf_path.split("/")[-1].replace(".pdf", "")  # Extract name from filename

        # Load PDF and split content
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        
        # Preprocess and split the document for manageable chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        # Create embeddings for document chunks
        embeddings = [embedding_model.embed_documents(doc.page_content) for doc in split_docs]

        # Concatenate processed resume text for prompt
        resume_text = " ".join([doc.page_content for doc in split_docs])

        # Generate questions with follow-ups in a single request
        output = question_chain.invoke({"candidate_resume": resume_text, "job_description": job_description})
        output_content = re.sub(r"^```json|```$", "", output.content, flags=re.MULTILINE).strip()
        questions = json.loads(output_content)  # Parse the JSON-like output

        # Populate the JSON structure
        interview_json[candidate_name] = {
            "questions": questions
        }

    return interview_json

# Paths to candidate PDF resumes and the job description text
# pdf_paths = ["Dhairya Arora.pdf"]
folder_path = "resumes"  # Replace with your folder path
pdf_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

job_description = "Develops scalable backend systems with expertise in Python and database management."

# Generate interview questions and save to JSON
interview_data = generate_interview_json(pdf_paths, job_description)

with open("interview_questions.json", "w") as file:
    json.dump(interview_data, file, indent=4)

print("Interview questions generated and saved to interview_questions.json.")
