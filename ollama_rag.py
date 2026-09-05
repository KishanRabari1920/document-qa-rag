import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama



# 1. Load PDF

#for single document (just for testing)
# pdf_path = "document.pdf"
# loader = PyPDFLoader(path)

path = "data"
docs = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        continue
    docs.extend(loader.load())

# print("PDF loaded successfully!")
print(f"Document Loaded Successfully!")
print(f"Number of pages: {len(docs)}")



# 2. Split PDF into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")



# 3. Create FREE local embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")



# 4. Create FAISS Vector Database

db = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created!")



# 5. Create FREE LOCAL LLM

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("Local LLM loaded!")


# 6. Ask questions continuously

while True:

    question = input(
        "\nAsk your question (type 'exit' to stop): "
    )

    if question.lower() == "exit":
        print("RAG stopped.")
        break


    # 7. Search relevant chunks
    
    results = db.similarity_search(
        question,
        k=6
    )

    
    # 8. Create context

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    
    # 9. Create prompt
    
    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information
provided in the document context below.

If the answer is not available in the context,
say:

"I don't know based on the provided document."

Document Context:
{context}

Question:
{question}

Answer:
"""

    # 10. Ask local LLM

    response = llm.invoke(prompt)

    # 11. Print answer
    
    print("\nAnswer:")
    print(response.content)