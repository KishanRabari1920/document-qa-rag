# ADVANCED RAG
# FAISS + BM25 HYBRID RETRIEVAL

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama



# 1. LOAD DOCUMENT
pdf_path = "langchain/documents/Unit 1_ Introduction to Machine Learning (Machine Learning - AIM5021C).pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Number of pages: {len(documents)}")

print("\nFirst 100 characters:")
print(documents[0].page_content[:100])

print("\nMetadata:")
print(documents[0].metadata)



# 2. CHUNKING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"\nNumber of chunks: {len(chunks)}")



# 3. ADD CHUNK METADATA
for i, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = i

    if "source" not in chunk.metadata:
        chunk.metadata["source"] = pdf_path


print("\nExample chunk metadata:")
print(chunks[0].metadata)



# 4. DENSE EMBEDDINGS
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded")



# 5. FAISS VECTOR DATABASE
vectorDB = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)

print("FAISS Vector DB created")



# 6. BM25 RETRIEVER
bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = 6

print("BM25 retriever created")



# 7. LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0.3
)

print("LLM loaded")



# 8. HYBRID RETRIEVAL FUNCTION
def hybrid_search(query, dense_k=6, sparse_k=6):


    # Dense Retrieval
    dense_docs = vectorDB.similarity_search(
        query,
        k=dense_k
    )

    # Sparse Retrieval - BM25
    bm25_retriever.k = sparse_k

    sparse_docs = bm25_retriever.invoke(query)

    # Combine Results
    combined_docs = dense_docs + sparse_docs

    # Remove Duplicate Chunks
    unique_docs = {}

    for doc in combined_docs:

        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id not in unique_docs:
            unique_docs[chunk_id] = doc


    return list(unique_docs.values())



# 9. QUESTION LOOP
while True:

    user_question = input(
        "\nAsk your question: "
    )

    if user_question.lower() == "exit":

        print("Goodbye! RAG has stopped.")
        break


    # 10. HYBRID RETRIEVAL
    retrieved_docs = hybrid_search(
        user_question,
        dense_k=6,
        sparse_k=6
    )


    # 11. DISPLAY RETRIEVED CHUNKS
    print("\nRetrieved chunks:")

    for i, doc in enumerate(retrieved_docs):

        print(f"\n--- Chunk {i + 1} ---")

        print(
            doc.page_content[:300]
        )

        print(
            "Metadata:",
            doc.metadata
        )



    # 12. LIMIT CONTEXT
    retrieved_docs = retrieved_docs[:8]

    # 13. CREATE CONTEXT
    context = "\n\n".join(
        f"""
        [Source: {doc.metadata.get("source")}
         Page: {doc.metadata.get("page")}
         Chunk: {doc.metadata.get("chunk_id")}]

        {doc.page_content}
        """
        for doc in retrieved_docs
    )


    # 14. PROMPT
    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

Rules:

1. Do not invent information.
2. If the answer is not present in the context,
   say:

   "I don't know based on the provided document."

3. Answer in English by default.
4. If the user asks for another language,
   answer in that language.
5. Give a concise and clear answer.
6. When possible, mention the source/page.

CONTEXT:

{context}


QUESTION:

{user_question}


ANSWER:
"""



    # 15. GENERATE ANSWER
    response = llm.invoke(prompt)

    # 16. FINAL ANSWER
    print("\nANSWER:")
    print(response.content)

    print("-" * 60)