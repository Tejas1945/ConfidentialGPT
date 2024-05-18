#import bitsandbytes
#from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
#from chromadb.utils import embedding_functions
#from langchain.vectorstores.chroma import Chroma
#from get_embedding_function import get_embedding_function
#from chromadb.utils import embedding_functions

import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from get_embedding_function import get_embedding_function


"""from langchain_community.embeddings import GPT4AllEmbeddings
def get_embedding_function():
    embeddings = GPT4AllEmbeddings()
    return embeddings"""

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
you are an assistant that answers in brief and only based on the context provided. The context is as follows:

{context}

---

The question is: {question}
"""

"""def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_llm(query_text)"""

def main(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    model = Ollama(model="llama2")
       
    response_text = model.invoke(prompt, temperature=0.3)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    #print(formatted_response)
    print(response_text)
    #print(type(response_text))
    sys.stdout.flush()
    return response_text

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        main(user_input)

"""
from sentence_transformers import SentenceTransformer

def get_embedding_function():
    model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    return model
"""

"""
def query_llm(query_text: str):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    prompts = [prompt]
    #model = Ollama(model="llama2")
    #response_text = model.invoke(prompt)
    


    # Code to inference Hermes with HF Transformers
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Pro-Llama-3-8B', trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
        "NousResearch/Hermes-2-Pro-Llama-3-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=False,
        load_in_4bit=True,
        use_flash_attention_2=True
    )


    for chat in prompts:
        print(chat)
        input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
        generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        print(f"Response: {response}")

    

    model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B" #"meta-llama/Meta-Llama-3-8B"

    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    response_text = pipeline(prompt)
    print(type(response_text))
    print (response_text)  
    
    model_path="capybarahermes-2.5-mistral-7b.Q5_K_S.gguf"
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    llm = Llama(
    model_path,  # Download the model file first
    n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
    #n_threads=8,             The number of CPU threads to use, tailor to your system and the resulting performance
    #n_gpu_layers=0          The number of layers to offload to GPU, if you have GPU acceleration available
    )

    # Simple inference example
    response_text = llm(
    prompt,          #"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant", # Prompt
    max_tokens=512,  # Generate up to 512 tokens
    #stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=True        # Whether to echo the prompt
    )

    # Chat Completion API

    llm = Llama(model_path, chat_format="llama-2")  # Set chat_format according to the model you are using
    
    llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are a story writing assistant."},
            {
                "role": "user",
                "content": "Write a story about llamas."
            }
        ]
    )
    
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text
"""



"""
GPU support using cuda
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python 
"""