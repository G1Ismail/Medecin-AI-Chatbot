from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def loadPdfDoc(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def textSplit(extractedData):
    textSpitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=20)
    textChunks = textSpitter.split_documents(extractedData)
    return textChunks

def downloadHuggingFace_Embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings