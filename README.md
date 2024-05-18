# ConfidentialGPT
### Setup:
1. Download [Ollama for windows] (https://ollama.com/download)
>To install llama2 llm model run command in terminal  ollama run llama2
>Model for embeddings generation run command in terminal  ollama pull all-minilm
2. Create a virtual environment suppose '.venv' is the virtual environment file name:
<clipboard-copy for="blob-path" class="btn btn-sm BtnGroup-item"> python -m venv .venv </clipboard-copy> <div id="blob-path">src/index.js</div>
3. To activate .venv run command in terminal
<clipboard-copy for="blob-path" class="btn btn-sm BtnGroup-item"> .venv/Scripts/activate </clipboard-copy> <div id="blob-path">src/index.js</div>
4. Install the requirements.txt file
<clipboard-copy for="blob-path" class="btn btn-sm BtnGroup-item"> pip install -r requirements.txt </clipboard-copy> <div id="blob-path">src/index.js</div>
### One mode thing and then you are good to go.
1. Change the location of data folder in create_vectorDB.py
