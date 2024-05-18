import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
import subprocess
import sys, os

def list_files(directory):
    files = os.listdir(directory)
    return files

def handle_userinput(user_question):
    with st.spinner("Processing"):
        # Run llm.py and capture the output
        result = subprocess.run([sys.executable, "llm.py", user_question], capture_output=True, text=True)
        # Check if the process ran successfully
        if result.returncode == 0:
            st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            # Display the output
            st.write(bot_template.replace("{{MSG}}", result.stdout), unsafe_allow_html=True)
        else:
            # Display error message if there was a problem running llm.py
            st.error("An error occurred while processing your question.")
            # Print stderr if available
            st.error(result.stderr)
            # Print stdout for debugging
            st.error(result.stdout)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Data")
        folder_path = r"C:\Users\prcsc\Documents\ConfGPT_trial\data"
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = list_files(folder_path)
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(folder_path, file)
                    # Convert file path to a file URL
                    file_url = file_path.replace('\\', '/')
                    st.sidebar.markdown(f"[{file}]({file_url})")
        if st.button("Process"):
            with st.spinner("Processing"):
                msg = subprocess.run([sys.executable, "create_vectorDB.py", "reset"], capture_output=True, text=True)
                st.write(msg.stdout)


if __name__ == '__main__':
    main()
