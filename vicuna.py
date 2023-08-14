import os
import streamlit as st
from streamlit_chat import message
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from sentence_transformers import SentenceTransformer
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from langchain import LLMChain, PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

model = SentenceTransformer('all-MiniLM-L6-v2')
embed_st = datetime.now()

# Define a custom embedding object
class CustomEmbedding:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, query):
        query_embedding = self.model.encode([query])
        return query_embedding.tolist()[0]

index = {"15cj01!.docx": "https://acsicorp.sharepoint.com/:w:/r/sites/NissanNorthAmerica/Shared%20Documents/General/Demo/tsb_demo_samples_3/15cj01!.docx?d=wf561f7195973431790b24a9b9f1e1c68&csf=1&web=1&e=U72dkF",
         "198.docx": "https://acsicorp.sharepoint.com/:w:/r/sites/NissanNorthAmerica/Shared%20Documents/General/Demo/tsb_demo_samples_3/198.docx?d=w33d744f015184185a892571f1a3b06d0&csf=1&web=1&e=ClEkjm",
         "236.docx": "https://acsicorp.sharepoint.com/:w:/r/sites/NissanNorthAmerica/Shared%20Documents/General/Demo/tsb_demo_samples_3/236.docx?d=w6c0e0ddf9c7c420b878ecfbb99b38771&csf=1&web=1&e=73iR60",
         "266.docx": "https://acsicorp.sharepoint.com/:b:/r/sites/NissanNorthAmerica/Shared%20Documents/General/tsb_demo_samples/acura/266.pdf?csf=1&web=1&e=PbkGjW",
         "728.docx": "https://acsicorp.sharepoint.com/:w:/r/sites/NissanNorthAmerica/Shared%20Documents/General/Demo/tsb_demo_samples_3/728.docx?d=waa06d49d087548a1baa07f691f2e3b92&csf=1&web=1&e=pQ8uBT",
         "914.docx": "https://acsicorp.sharepoint.com/:w:/r/sites/NissanNorthAmerica/Shared%20Documents/General/Demo/tsb_demo_samples_3/914.docx?d=w2f71e152ec2f4831a1e7a1baff2c6be9&csf=1&web=1&e=CBOFbR",
         "TSB Video Embed Sample.docx": "https://acsicorp.sharepoint.com/:w:/r/sites/NissanNorthAmerica/Shared%20Documents/General/Demo/tsb_demo_samples_3/TSB%20Video%20Embed%20Sample.docx?d=wd213c81bd0c843a0bf0fa5342661ed3d&csf=1&web=1&e=PFW5KQ"
         }


embedding = CustomEmbedding(model)

# Create Persist Directory
persist_directory = 'db'
db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

docsearch = db.as_retriever(search_kwargs={"k": 3})


# create the embedding of query as well then store it to croma db and pass it to language model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_LS = datetime.now()
llm = LlamaCpp(
        model_path="models/ggml-vic13b-q8_0.bin",
        callback_manager=callback_manager,
        n_gpu_layers=10,
        verbose=True,
        n_ctx=2048,
        n_batch=512
    )



template = """ You are a Nissan Chatbot, to help users with their queries.
Given the following extracted parts of a long document and a question, create a helpful answer. 
If the answer is not found in document then say the requested information is not available.

{summaries}

{chat_history}
Employee: {question}
Nissan Chatbot:"""


prompt = PromptTemplate(
    input_variables=["chat_history", "question","summaries"], 
    template=template
)

if 'memory' not in st.session_state:
    st.session_state.memory=ConversationBufferWindowMemory(input_key="question",memory_key="chat_history",k=3)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
    

# Setting page title and header
st.set_page_config(page_title="Nissan Conversation Demo", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Nissan Conversation Demo</h1>", unsafe_allow_html=True)



# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'links' not in st.session_state:
    st.session_state['links']=[]
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")


# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['links']=[]
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


# generate a response
def generate_response(query):
    
    if 'question_chain' not in st.session_state: 
        st.session_state.question_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    if 'source_chain' not in st.session_state: 
        st.session_state.source_chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff",prompt=prompt)

    if 'combine_chain' not in st.session_state: 
        st.session_state.combine_chain =  ConversationalRetrievalChain(retriever=docsearch, question_generator= st.session_state.question_chain, combine_docs_chain=st.session_state.source_chain, return_source_documents=True)


    start = datetime.now()
    
    c = st.session_state.combine_chain({"question": query, "chat_history": st.session_state.chat_history})
    
   
            
    
    result = {
            'query' : query,
            'answer' : c['answer']
        }

    st.session_state.chat_history = [(query, result["answer"])]

    st.session_state['messages'].append({"role": "user", "content": query})
    response = c['answer'] 
    
    end = datetime.now()
    time ="Response Time: " + str((end-start).total_seconds()) + " s"
    st.write(time)

    st.session_state['messages'].append({"role": "assistant", "content": response})

    docs = docsearch.get_relevant_documents(query)


    links = []

    for i in c["source_documents"]:
        meta = i.metadata["source"]
        filename = os.path.basename(meta)
        print(filename)
        filename = filename.rsplit(".", 1)[0]
        pdf_link = index[filename]
        source_link = "Source:[{document}]({link})".format(document={filename},link=pdf_link)
        links.append(source_link)

    # for i in docs:
    #     print(len(i.page_content))
    #     pdf = str(i.metadata["source"])
    #     pdf = pdf.rstrip(".txt")
    #     pdf = pdf.split("\\")
    #     #st.write(pdf[1])
    #     pdf_link = index[pdf[1]]
    #     pdf_link = "Source:[{document}]({link})".format(document={pdf[1]},link=pdf_link)
    #     print(pdf_link)
    #     links.append(pdf_link)

    set_links = set(links) 
    list_res = (list(set_links))
    
    
    return response,list_res,time


message("Hi, How can I assist you with NISSAN/INFINITI questions?", key="-1")
# container for chat history
response_container = st.container()
spinner_container = st.container()
# container for text box
container = st.container()
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input:
        with spinner_container:
            with st.spinner("I'm working on your request..."):
                output,links,time = generate_response(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
            #display_link(links)
                st.session_state['links'].append(links)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            #display_link(links)
            for i in st.session_state["links"][i]:
                response_container.write(i)
