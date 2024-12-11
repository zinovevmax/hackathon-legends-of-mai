import os
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_text_files(directory):
    print('loading documents')
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
                print(f"{filename} ready")
    return documents


documents = load_text_files("data")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(len(docs))
if len(docs) == 0:
    raise ValueError("no split ", len(docs))
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("creating vector store")
vectorstore = Chroma.from_documents(docs, hf_embeddings, persist_directory="chroma_storage")
print("Vectorstore saved successfully.")

# Получение всех данных: текстов, метаданных и векторов
data = vectorstore._collection.get()

# Соединение текстов с векторами
for i in range(len(data["documents"])):
    print(f"Chunk {i}: {data['documents'][i]}")
    print(f"Metadata: {data['metadatas'][i]}")
    #print(f"Vector: {data['embeddings'][i]}")
    print("-" * 50)

