import os
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_text_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents


documents = load_text_files("data")

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(documents, hf_embeddings, persist_directory="chroma_storage")

print("Vectorstore saved successfully.")

query = "what is Agents in langchain"
results = vectorstore.similarity_search(query, k=3)

for result in results:
    print(f"Content: {result.page_content}")
    print(f"Source: {result.metadata['source']}")
    print("-" * 50)
