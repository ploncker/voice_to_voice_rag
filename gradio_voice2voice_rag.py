
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 08:14:42 2023

@author: ross
"""

from dotenv import dotenv_values, find_dotenv
config = dotenv_values(find_dotenv())

import gradio as gr
from gradio_pdf import PDF

import time
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os


from pyht.client import Client, TTSOptions
from pyht.protos import api_pb2

import numpy as np
import speech_recognition as sr
import io
from scipy.io.wavfile import write
from PyPDF2 import PdfReader
import shutil

system_message = {"role": "system", "content": "Your name is Emma. You are a helpful assistant."}
api_key = None
docsearch = None 

def transcribe(audio):
    
    sample_rate, y = audio
    y = np.int16(y/np.max(np.abs(y)) * 32767)
    byte_io = io.BytesIO()
    write(byte_io, sample_rate, y)
    result_bytes = byte_io.read()
    
    audio_data = sr.AudioData(result_bytes, sample_rate, 2)
    r = sr.Recognizer()
    text = r.recognize_google(audio_data)
    return text


def read_file(filen):
    print(filen)
    reader = PdfReader(filen)
    raw_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text.append(text)
            
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ".", ",", " ", ],
        chunk_size = 500,
        chunk_overlap  = 100,
        length_function = len,
    )
    
    page_docs = [Document(page_content=page) for page in raw_text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i+1
    doc_chunks = []
    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content = chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata['source'] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    
    return doc_chunks


def embed_docs(filen):
    doc_chunks = read_file(filen)
    global embeddings 
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    global docsearch 
    docsearch = FAISS.from_documents(doc_chunks, embeddings)
    return "Done"


def tts(text):
    # Setup the client
    voice = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json"
    quality = "faster"
    #use_async = False
    client = Client(config['PLAYHT_USERID'], config['PLAYHT_KEY'])

    # Set the speech options
    options = TTSOptions(voice=voice, format=api_pb2.FORMAT_WAV, quality=quality)
    
    sr = 24000
    buff_size = 400*sr
    ptr = 0
    start_time = time.time()
    buffer = np.empty(buff_size, np.int16)
    for i, chunk in enumerate(client.tts(text, options)):
        if i == 0:
            start_time = time.time()
            continue  # Drop the first response, we don't want a header.
        elif i == 1:
            print("First audio byte received in:", time.time() - start_time)
        for sample in np.frombuffer(chunk, np.int16):
            buffer[ptr] = sample
            ptr += 1
    approx_run_size = ptr
    buffer2 = buffer[:approx_run_size]
    
    return (sr, buffer2)
        
def set_openai_api_key(api_key: str):
    """
    Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
def user(user_message, history):
    return "", history + [[user_message, None]]

    
def bot(history, messages_history):
    user_message = history[-1][0]
    bot_message, messages_history = ask_gpt(user_message, messages_history)
    emma = tts(bot_message)
    messages_history += [{"role": "assistant", "content": bot_message}]
    history[-1][1] = bot_message
    time.sleep(1)
    return history, messages_history, emma

def ask_gpt(message, messages_history):
    docs = docsearch.similarity_search(message, k=8)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    output = chain.run(input_documents=docs, question=message)
    messages_history += [{"role": "user", "content": output}]
    return output, messages_history
    
def init_history(messages_history):
    messages_history = []
    messages_history += [system_message]
    return messages_history

def upload_file(fileobj):    
    UPLOAD_FOLDER = "./data"    
    if not os.path.exists(UPLOAD_FOLDER):    
        os.mkdir(UPLOAD_FOLDER) 
    path = UPLOAD_FOLDER + '/' +os.path.basename(fileobj) 
    shutil.copy(fileobj, UPLOAD_FOLDER)    
    #gr.Info("File Uploaded!!!")   
    return path
    
with gr.Blocks() as demo:
    
    with gr.Row():
        with gr.Column(scale=2, min_width=600):
            pdf = PDF(label="Upload a PDF", scale = 1, min_width=800, interactive=True)
            name = gr.Textbox(type='text', visible=True)
            #pdf.upload(lambda f: f, pdf, name)
            #pdf.upload(lambda fn: fn, pdf, name)
            pdf.upload(upload_file, pdf, name)
            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key (sk-...)",
                show_label=False,
                lines=1,
                type="password",
            )
            btn = gr.Button("Embed pdf", scale=0)
            status_box = gr.Textbox()
            btn.click(embed_docs, name, status_box, show_progress=True)
            
            
            
        with gr.Column(scale=2, min_width=600):
            inputs = [gr.Audio(sources=["microphone"])]
            btn = gr.Button("Transcribe", scale=0)
            msg = gr.Textbox()
            btn.click(transcribe, inputs, msg)
            submit_btn = gr.Button("Submit")
            
            chatbot = gr.Chatbot()
            myaudio = gr.Audio(autoplay=True)
            
            clear = gr.Button("Clear")
    
        
    state = gr.State([])     

    openai_api_key_textbox.change(
            set_openai_api_key,
            inputs=[openai_api_key_textbox],
        )

    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, state ], [chatbot, state, myaudio]
    )

    clear.click(lambda: None, None, chatbot, queue=False).success(init_history, [state], [state])

if __name__ == "__main__":
    demo.launch(debug=True)
