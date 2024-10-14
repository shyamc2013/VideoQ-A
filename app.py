import streamlit as st
# from langchain.documnet_loader import DirectoryLoader
# from langchain.loaders import DirectoryLoader
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader        #Load DOCX file using docx2txt and chunks at character level
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_openai import OpenAI
from docx import Document       #used for creating and updating Ms Word documents
# from langchain.vectorstores import Chroma
import os
# from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.chains import ConversationChain, RetrievalQA
# from watsonxlangchain import LangChainInterface
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
import time
from utils import *
import moviepy.editor as mp      #library for video editing
import azure.cognitiveservices.speech as speechsdk  #accessing Microsoft's Azure Cognitive Services Speech SDK
import wave


os.environ["OPENAI_API_KEY"] = "b7d5aa82d15a4b99a1c730f681ec2bbc"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://hmh-digitalhub-azure-openai.openai.azure.com/"
os.environ["CHAT_MODEL"] = "gpt-35-turbo"
os.environ["CHAT_MODEL_DEPLOYMENT_NAME"] = "gpt-35-turbo"
os.environ["EMBEDDINGS_MODEL"] = "text-embedding-ada-002"
os.environ["EMBEDDINGS_MODEL_DEPLOYMENT_NAME"] = "text-embedding-ada-002"

llm = AzureChatOpenAI(  
            model_name ="gpt-35-turbo",
            deployment_name= "gpt-35-turbo",
            temperature=0,
            openai_api_version ="2023-07-01-preview",
            openai_api_key="b7d5aa82d15a4b99a1c730f681ec2bbc",
            openai_api_base="https://hmh-digitalhub-azure-openai.openai.azure.com/"
        )


def setup_speech_recognizer(file) -> speechsdk.SpeechRecognizer:
    print("Setting up speech recognition ...")
    try:
        #configuration settings required to use Azure's Speech Service
        speech_config = speechsdk.SpeechConfig(
            subscription= "a991d83e4f7d4861bd6707133c9db66a",
            region= "eastus",
            speech_recognition_language= "en-US"
        )

        audio_config = speechsdk.AudioConfig(filename=f"Audio_file/{file}.wav")      #this object specifies the source of audio
        #speech_recognizer object is responsible for performing the speech recogntion
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)  
    except Exception as e:
        print(RED + "Could not setup speech service" + ENDC)
        print(e)
        sys.exit(1)
    else:
        return speech_recognizer

def recognize(speech_recognizer: speechsdk.SpeechRecognizer) -> list:
    print("Starting speech recognition ...\n")
    done = False
    all_results = list()

    def stop_cb(evt):
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True

        if evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            cancellation_reason = cancellation_details.reason
            print()
            if cancellation_reason == speechsdk.CancellationReason.Error:
                print(RED + "=== ERROR ===" + ENDC)
                print(f"Error code: {cancellation_details.error_code}")
                print(f"Error details: {cancellation_details.error_details}")
            elif cancellation_reason == speechsdk.CancellationReason.CancelledByUser:
                print(RED + "=== CANCELED BY USER ===" + ENDC)
            elif cancellation_reason == speechsdk.CancellationReason.EndOfStream:
                print(GREEN + "=== SUCCESS ===" + ENDC)

    #saving the final recognized transcription to the 'all_results' list
    def handle_final_result(evt):
        nonlocal all_results
        all_results.append(evt.result.text)

    # connect callbacks to the events fired by the speech recognizer

    #'recognized' event is triggered when the speech recognizer has successfully recognized a spoken phrase and has a final result (transcription) available.
    speech_recognizer.recognized.connect(handle_final_result)
    #'recognizing' event is fired when the speech recognizer is processing spoken audio but hasn't yet completed the recognition process
    speech_recognizer.recognizing.connect(lambda evt: print(f"RECOGNIZING: {evt}"))
    speech_recognizer.recognized.connect(lambda evt: print(f"RECOGNIZED: {evt}"))
    #'session_started' and 'session_stopped' event are fired when a new speech recognition session begins and ends respectively
    speech_recognizer.session_started.connect(lambda evt: print(f"SESSION STARTED: {evt}"))
    speech_recognizer.session_stopped.connect(lambda evt: print(f"SESSION STOPPED: {evt}"))
    #'canceled' event is fired when the speech recognition process is canceled, either by the user or due to an error
    speech_recognizer.canceled.connect(lambda evt: print(f"CANCELED: {evt}"))

    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(5)                      #halts the execution of the program for 5 seconds for speech recognizer to do its work

    return all_results

def create_word_document(text_results: list[str], output_file_path: str) -> None:
    try:
        # Create a new Word document
        doc = Document()

        # Add each text result as a paragraph to the document
        for result in text_results:
            doc.add_paragraph(result)

        # Save the document
        doc.save(output_file_path)
        st.write(f"Word document saved to: {output_file_path}")
    except Exception as e:
        st.write(f"Could not create Word document: {e}")


#Decorator to cache functions that return data
@st.cache_data
def initiate_process(uploaded_file):
    temp_dir = 'Video_file'
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    #uploaded video file is saved to the location specified in temp_file_path
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    

    #splitting 'file.mp4' into 'file' and 'mp4'
    file_name, _ = os.path.splitext(uploaded_file.name)
    VIDEO_URL = temp_file_path
    # st.write(f"File uploaded successfully! Filename: {file_name}")

    with st.status("Generating audio file..."):
        start_time = time.time()
        clip = mp.VideoFileClip(f"{temp_file_path}")            #opening the video file
        clip.audio.write_audiofile(f"Audio_file/{file_name}.wav")      #extracting audio from the video and saving it in audio file
        elapsed_time = time.time() - start_time
        st.write("Audio file created!!")
        st.write("Time taken for generating audio:", elapsed_time, "seconds")

    with st.status("Generating transcription..."):
        start_time = time.time()

        #extracting the transcripts from the audio using Azure Cognitive Services Speech SDK 
        speech_recognizer = setup_speech_recognizer(file_name)
        text_results = recognize(speech_recognizer)

        output_file_path = f"transcript/{file_name}.docx"
        create_word_document(text_results, output_file_path)   #saving the recognized transcript to the location output_file_path
        elapsed_time = time.time() - start_time
        st.write("Time taken for transcription:", elapsed_time, "seconds")
    return file_name,temp_file_path

"""execution of the file starts from here"""
#upload the video file
uploaded_file = st.file_uploader("Choose a file", type=["mp4"])

if uploaded_file is not None:
    file_name,temp_file_path = initiate_process(uploaded_file)
    # temp_dir = 'Video_file'
    # temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    # with open(temp_file_path, "wb") as temp_file:
    #     temp_file.write(uploaded_file.read())
    

    
    # file_name, _ = os.path.splitext(uploaded_file.name)
    VIDEO_URL = temp_file_path
    # # st.write(f"File uploaded successfully! Filename: {file_name}")

    # with st.status("Generating audio file..."):
    #     start_time = time.time()
    #     clip = mp.VideoFileClip(f"{temp_file_path}")
    #     clip.audio.write_audiofile(f"Audio_file/{file_name}.wav")
    #     elapsed_time = time.time() - start_time
    #     st.write("Audio file created!!")
    #     st.write("Time taken for generating audio:", elapsed_time, "seconds")

    # with st.status("Generating transcription..."):
    #     start_time = time.time()
    #     speech_recognizer = setup_speech_recognizer(file_name)
    #     text_results = recognize(speech_recognizer)

    #     output_file_path = f"transcript/{file_name}.docx"
    #     create_word_document(text_results, output_file_path)
    #     elapsed_time = time.time() - start_time
    #     st.write("Time taken for transcription:", elapsed_time, "seconds")


    # if st.checkbox("Ask query"):


    documents = []
    loader = Docx2txtLoader(f"transcript/{file_name}.docx")    
    documents.extend(loader.load())   #loader.load() gives a list of Document objects

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    st.write(len(chunked_documents))


    # st.title("ChatBot")

    #display a video player
    st.video(VIDEO_URL)

    if 'messages' not in st.session_state:
        st.session_state.messages =[]
        
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")


    embeddings = AzureOpenAIEmbeddings()

    # load it into Chroma
    # db = Chroma.from_documents(docs, embedding_function)
    document_search = FAISS.from_documents(chunked_documents, embeddings)
    # st.write(len())
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever(k = 6))
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=document_search.as_retriever())

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        response = chain(prompt)

        st.chat_message('assistant').markdown(response['result'])
        st.session_state.messages.append(
            {'role':'assistant','content':response['result']}
        )
