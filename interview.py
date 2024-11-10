from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, VertexAI


from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
import json

import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile


load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm = VertexAI(model_name="gemini-1.0-pro-001", safety_settings=safety_settings)


model = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)

embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

store = {}

faiss_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Define a dictionary to store FAISS indices for each session
faiss_indices = {}
faiss_stores = {}

def get_faiss_index(session_id: str):
    if session_id not in faiss_indices:
        dimension = 384  # Dimension of the embeddings
        faiss_indices[session_id] = faiss.IndexFlatL2(dimension)
        faiss_stores[session_id] = {}
    return faiss_indices[session_id], faiss_stores[session_id]


def store_in_faiss(data_text: str, session_id: str):
    index, faiss_store = get_faiss_index(session_id)

    embedding_vector = embedding.embed_query(data_text)
    embedding_vector = np.array(embedding_vector).reshape(1, -1)
    index.add(embedding_vector)
    faiss_store[data_text] = embedding_vector
    print(f"Added to FAISS for session {session_id}: {data_text}")


def search_faiss(query_text, session_id):
    index, _ = get_faiss_index(session_id)

    query_vector = embedding.embed_query(query_text)
    query_vector = np.array(query_vector).reshape(1, -1)

    distances, indices = index.search(query_vector, k=5)

    if indices.size == 0 or indices[0][0] == -1:
        return []

    results = []
    faiss_store = faiss_stores.get(session_id, {})
    for idx in indices[0]:
        if 0 <= idx < len(faiss_store):
            results.append(list(faiss_store.keys())[idx])

    return results

def extract_text_from_pdf_file(file: UploadFile) -> str:
    """
    Extract text from an uploaded PDF file using PyPDFLoader or similar.
    """
    tmp_dir = tempfile.gettempdir()
    tmp_file_path = os.path.join(tmp_dir, file.filename)

    with open(tmp_file_path, "wb") as tmp_file:
        tmp_file.write(file.file.read())

    loader = PyPDFLoader(file_path=tmp_file_path)
    docs = loader.load()
    
    # Preprocess and split the document for manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    resume_text = " ".join([doc.page_content for doc in split_docs])

    return resume_text

def getAllContext(session_id):
    index, _ = get_faiss_index(session_id)
    all_context = []
    faiss_store = faiss_stores.get(session_id, {})
    for text in faiss_store.keys():
        all_context.append(text)
    return all_context


# Define the prompt with conversational context
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''You are an HR assistant, AI-HR, conducting a job interview based on a candidate's Resume and Job Description (JD).
                    Greet the candidate in the first message of the session, then ask questions relevant to the JD and Resume.
                    Maintain a conversational flow, ensuring you do not repeat any questions.
                    If the candidate's qualifications or skills match a part of the JD, ask follow-up questions for further details.
                    Use the Resume and JD information to make your questions precise and avoid unnecessary queries.
                    Thank the candidate politely when they respond or complete the interview.'''),
        ('system', 'This is the context of user from Database: {faiss_context}'),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

app = FastAPI(title="AI-HR", version="1.0.0", description="AI-HR Interview API")

@app.post("/init")
async def initialize_interview(
    JD: str = Form(...),
    session_id: str = Form(...),
    resume: UploadFile = File(...),
):
    """
    Initialize interview with JD and resume PDF. Extract text from the resume and store embeddings in FAISS.
    """
    if not JD or not session_id or not resume:
        raise HTTPException(status_code=400, detail="JD, session_id, and resume are required fields.")

    try:
        # Extract text from the uploaded resume PDF
        resume_text = extract_text_from_pdf_file(resume)

        # Store JD and Resume context in FAISS
        store_in_faiss(JD, session_id)
        store_in_faiss(resume_text, session_id)

        return {"response": f"JD and Resume initialized and stored for session {session_id}"}

    except Exception as e:
        return {"response": f"Error during initialization: {str(e)}"}
    

@app.post("/interview")
async def invoke_interview(request: Request):
    data = await request.json()

    input_data = data.get("input", {})
    if not input_data:
        return {"response": "Invalid input format. No content found."}

    content = input_data.get("content", "")
    session_id = input_data.get("session_id", "")

    # Search for relevant context in FAISS
    faiss_results = search_faiss(content, session_id)
    faiss_context = faiss_results if faiss_results else "No relevant context found."

    try:
        response = with_message_history.invoke(
            input={
                "input": [HumanMessage(content=content)],
                "faiss_context": faiss_context
            },
            config={"configurable": {"session_id": session_id}}
        )
    except Exception as e:
        response = str(e)

    return {"response": response}


evaluation_prompt_template = """
You are an AI assistant specialized in evaluating job interviews. Based solely on the candidate's responses during the interview session, evaluate their performance on the following parameters. For each parameter, assign a score out of 100 and provide a brief explanation for the score, using the Job Description (JD) and Candidate's Resume for contextual reference only. Do not consider the JD and Resume as criteria for scoring.
Terminate the interview with a goobye message when you think most of the things relevant to the JD and resume have been covered, you can check this from the message history provided.

### Interview Message History:
{message_history}

### Evaluation Parameters:
1. **Correctness of Answer**: How accurately does the candidate answer the questions?
2. **Completeness of Answer**: How thorough and detailed are the candidate's responses?
3. **Knowledge of Own Projects in Resume**: How well does the candidate explain and demonstrate their projects listed in the resume?
4. **Communication Skills**: How effectively does the candidate communicate their ideas?
5. **Technical Skills**: How strong are the candidate's technical abilities relevant to the job?
6. **Problem-Solving Skills**: How proficient is the candidate in addressing and solving problems?
7. **Experience**: How well does the candidate's experience match the requirements of the JD?
8. **Candidate's Compatibility**: How suitable is the candidate for the role based on the JD?
9. **Skill Accuracy**: Out of all the skills mentioned in the resume, how many does the candidate effectively demonstrate during the interview?

### Instructions:
- Provide a JSON object with each parameter as a key.
- Even if you don't have all the information, provide a score based on the available context, you may give 0 if you think you have less information.
- Each key should map to an object containing:
  - `score`: An integer between 0 and 100.
  - `explanation`: A brief explanation for the score.

### Example Output:
{{
    "Correctness of Answer": {{
        "score": 85,
        "explanation": "The candidate answered most questions accurately but had minor inaccuracies in one technical detail."
    }},
    ...
}}
"""


# Initialize the LLM and Chain
evaluation_prompt = ChatPromptTemplate.from_template(
    template=evaluation_prompt_template,
    # input_variables=["jd_text", "resume_text", "message_history"]
)

evaluation_chain = evaluation_prompt | model

@app.post("/evaluate")
async def evaluate_candidate(session_id: str = Form(...)):
    """
    Evaluate the performance of the candidate based on the session history,
    JD, and resume context. The output includes a score out of 100 for each
    evaluation criterion.
    """
    # Retrieve chat history
    session_history = get_session_history(session_id)
    message_history = "\n".join([f"{msg.type}: {msg.content}" for msg in session_history.messages])
    context = getAllContext(session_id)

    if context == []:
        raise HTTPException(status_code=400, detail="JD and Resume context not found for the session.")
    # # Retrieve JD and Resume context from FAISS
    jd_text = context[0]
    resume_text = context[1]

    # print(f"JD: {jd_text}")
    # print(f"Resume: {resume_text}")
    print(message_history)
        
    if not jd_text or not resume_text:
        raise HTTPException(status_code=400, detail="JD and Resume context not found for the session.")
    
    try:
        # Generate evaluation using the LLM
        evaluation_output = evaluation_chain.invoke({
            "jd_text": jd_text,
            "resume_text": resume_text,
            "message_history": message_history
        })
        print(evaluation_output.content)
        content = evaluation_output.content.strip()
        start = content.find('```json')
        end = content.rfind('```')

        if start != -1 and end != -1 and end > start:
            json_content = content[start + 7:end].strip()  # +7 to skip past the "```json" part
            evaluation_metrics = json.loads(json_content)
        else:
            raise ValueError("Invalid response format: JSON block not found.")

        return JSONResponse(content=evaluation_metrics)
    
    except Exception as e:
        return {"response": f"Error during evaluation: {str(e)}"}


@app.get("/cron-job")
async def cron_job():
    return "CJ"


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8001)