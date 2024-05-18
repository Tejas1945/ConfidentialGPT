### ConfidentialGPT
#Setup:
Download Ollama for windows:
1.        https://ollama.com/download
In CMD run command "ollama run llama2" to install llama2 llm model and "ollama pull all-minilm" model for embeddings generation.
Create a virtual environment suppose '.venv' is the virtual environment file name:
2.        python -m venv .venv
To activate .venv run command in terminal
3.        .venv/Scripts/activate
Install the requirements.txt file
4.        pip install -r requirements.txt
# One mode thing and then you are good to go.
1. Change the location of data folder in create_vectorDB.py
