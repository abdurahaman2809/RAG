import os
from langchain_experimental.text_splitter import SemanticChunker
from vectorembed import embeddings

from langchain_chroma import Chroma 
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.rate_limiters import InMemoryRateLimiter

# Create a rate limiter to control API request rate
rate_limter = InMemoryRateLimiter(requests_per_second=0.05)

# Read the cleaned text data from file
with open('cleaned.txt', 'r', encoding='utf-8') as file:
    cleaned_text = file.read()

# Initialize the semantic chunker to split text into semantically meaningful chunks
chunker = SemanticChunker(embeddings=embeddings, breakpoint_threshold_amount=0.9)

# Split the cleaned text into chunks/documents
docs = chunker.create_documents([cleaned_text])
print(f"Total chunks created: {len(docs)}")

# for i in range(len(docs)):
#     print(f"Total chunks created: {docs[i].page_content}")

# Create a Chroma vector database from the document chunks
vector_db = Chroma.from_documents(
    embedding=embeddings,
    documents=docs, 
    persist_directory="chroma_db", 
    collection_name="my_collection"
)

# Get the OpenAI API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI chat model with rate limiting
llm = ChatOpenAI(
    api_key=openai_key,
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    rate_limiter=rate_limter
)

# Create a RetrievalQA chain that uses the LLM and the vector database retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

# Query the QA chain with a sample question
response = qa_chain.invoke({"query": "Engine capacity of Honda Accord?"})
print(f"Response Received is : {response['result']}")