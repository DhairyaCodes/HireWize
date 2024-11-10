import streamlit as st
import requests
import pandas as pd

# Set up API endpoint URLs
API_BASE_URL = "http://localhost:8001"  # Adjust if running on a different host or port
INIT_URL = f"{API_BASE_URL}/init"
INTERVIEW_URL = f"{API_BASE_URL}/interview"
EVALUATE_URL = f"{API_BASE_URL}/evaluate"

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Function to set page
def navigate_to(page):
    st.session_state['page'] = page

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.button("Interview Setup", on_click=lambda: navigate_to('setup'))
st.sidebar.button("Conduct Interview", on_click=lambda: navigate_to('interview'))
st.sidebar.button("Evaluation", on_click=lambda: navigate_to('evaluate'))



if st.session_state['page'] == 'interview':
    st.title("Conduct Interview")

    # Ensure session ID exists
    if 'session_id' in st.session_state and st.session_state['session_id']:
        # Display the chat messages (interview Q&A history)
        st.subheader("Interview Chat")
        for message in st.session_state["chat_history"]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        # User input for interview question
        interview_query = st.chat_input("Enter your next interview question...")
        
        if interview_query:
            # Append user question to chat history
            st.session_state["chat_history"].append({"role": "user", "content": interview_query})
            st.chat_message("user").write(interview_query)

            # Prepare data for API call to get AI response
            data = {
                "input": {
                    "content": interview_query,
                    "session_id": st.session_state["session_id"]
                }
            }
            response = requests.post(INTERVIEW_URL, json=data)
            
            if response.status_code == 200:
                # Extract AI response and add to chat history
                ai_response = response.json().get("response", "")
                st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
                st.chat_message("assistant").write(ai_response)
            else:
                st.error("Failed to get response from the interview API.")
    else:
        st.warning("Please start the interview from the initialization page first.")

            
elif st.session_state['page'] == 'evaluate':
    st.title("Evaluation")
    # Add evaluation code here
    # Ensure session ID exists
    if 'session_id' in st.session_state and st.session_state['session_id']:
        if st.button("Evaluate Interview"):
            # Send evaluation request
            data = {"session_id": st.session_state['session_id']}
            response = requests.post(EVALUATE_URL, data=data)
            
            if response.status_code == 200:
                evaluation = response.json()
                
                # Display evaluation results in a table format
                st.subheader("Evaluation Results")
                st.write("Here are the evaluation criteria and scores for the candidate:")

                # Use a table to present evaluation criteria, scores, and explanations
                table_data = []
                for criterion, result in evaluation.items():
                    score = result.get("score", "N/A")
                    explanation = result.get("explanation", "No explanation provided.")
                    table_data.append([criterion, score, explanation])

                # Display the results in a table-like format
                st.table(pd.DataFrame(table_data, columns=["Criterion", "Score", "Explanation"]))
                
            else:
                st.error("Evaluation failed. Please check the server.")
    else:
        st.warning("Please start the interview from the setup page first.")

else:
    st.title("Interview Setup")
    # Add setup code or form here

    # Input fields for Job Description, resume upload, and session ID
    JD = st.text_area("Job Description (JD)", "")
    resume_file = st.file_uploader("Upload Candidate's Resume (PDF)", type="pdf")
    session_id = st.text_input("Session ID")
    
    # Button to start interview
    if st.button("Start Interview"):
        if JD and resume_file and session_id:
            # Prepare data for API call
            files = {'resume': resume_file.getvalue()}
            data = {'JD': JD, 'session_id': session_id}
            
            # Call the API to initialize interview
            response = requests.post(INIT_URL, files=files, data=data)
            if response.status_code == 200:
                st.session_state['session_id'] = session_id
                st.session_state['chat_history'] = []
                st.success("Interview initialized successfully!")
            else:
                st.error("Failed to initialize interview. Please try again.")
        else:
            st.warning("Please provide all inputs: JD, Resume, and Session ID.")
