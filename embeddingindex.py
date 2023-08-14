from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import os

# Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

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


# os.chdir("C:/Users/joshua.dominic.ACS/Desktop/Embedding/")

loader = DirectoryLoader('./text_docs/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=70, separator="\n\n")
texts = text_splitter.split_documents(documents)
print("NO:ofChunks:",len(texts))
for i in texts:
    print(len(i.page_content))

embedding = CustomEmbedding(model)
persist_directory = 'db'

vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = None

    

print("Created VectoreStore")





