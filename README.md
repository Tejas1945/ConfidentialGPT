# ConfidentialGPT
### Setup:
1. Download [Ollama for windows] (https://ollama.com/download)
2. To install llama2 llm model run command in terminal  __ ollama run llama2 __
3. Model for embeddings generation run command in terminal  __ ollama pull all-minilm __
4. Create a virtual environment suppose '.venv' is the virtual environment file name. __ python -m venv .venv __
5. To activate '.venv' run command in terminal. __ .venv/Scripts/activate __
6. Install the 'requirements.txt' file. __ pip install -r requirements.txt __ 
### One mode thing and then you are good to go.
1. Change the location of data folder in create_vectorDB.py
