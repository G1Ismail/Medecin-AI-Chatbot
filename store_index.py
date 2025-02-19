# dieses script muss nur einmal laufen, also wenn in Pinecone database
# keine index gibt oder wenn die Datein sich geändert hatten:

import os
from dotenv import load_dotenv

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.functions import *

load_dotenv()

vektorDBIndexExists = 0
indexName = "medicin-chat-bot"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extractedData = loadPdfDoc(data='Data/')
textChunks = textSplit(extractedData=extractedData)
embeddings = downloadHuggingFace_Embeddings()

pc =  Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name=indexName, # Ich hatte ein Problem hier: nach lange Suche --> Losüng: keine Uppercases sind in Pinecone Indexes erlaubt
    dimension=384, # diese Dimension entspricht die Dimension vom HugginfFace Model, es ist sehr wichtig auf diese Dimension zu achten (Falls zukünftig anderes Model)
    metric="cosine", # andere metrics z.B: euclidean ... 
    spec=ServerlessSpec(
        # diese Specs sind die einzigen die gratis sind
        cloud="aws",
        region="us-east-1"
    )
)
#if (vektorDBexists == ):
    # 
    #
    
#else
    # 
    # 
    
docSearch = PineconeVectorStore.from_documents(
    documents=textChunks,
    index_name=indexName,
    embedding=embeddings
)

docSearch = PineconeVectorStore.from_existing_index(
    index_name=indexName,
    embedding=embeddings
)