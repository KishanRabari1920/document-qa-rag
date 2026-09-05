import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS



# 1. Load API Key

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in .env file")


# 2. Load PDF

pdf_path = 'document.pdf'

loader = PyPDFLoader(pdf_path)

docs = loader.load()

print(f"PDF loaded successfully!")
print(f"Number of pages: {len(docs)}")


# 3. Split PDF into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")


# 4. Create Embeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 5. Create FAISS Vector Database

db = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created!")


# 6. Create LLM

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=10
)


# 7. Ask questions continuously

while True:

    question = input("\nAsk your question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("RAG stopped.")
        break

    
    # 8. Search relevant chunks
    
    results = db.similarity_search(
        question,
        k=3
    )

    
    # 9. Create context from retrieved chunks
    
    context = "\n\n".join(
        doc.page_content
        for doc in results
    )


    # 10. Create prompt
    
    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information
provided in the PDF context below.

If the answer is not available in the context,
say:

"I don't know based on the provided document."

PDF Context:
{context}

Question:
{question}
"""


    # 11. Ask LLM
    
    response = llm.invoke(prompt)

    
    # 12. Print answer

    print("\nAnswer:")
    print(response.content)