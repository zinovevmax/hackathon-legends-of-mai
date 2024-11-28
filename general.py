from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="chroma_storage", embedding_function=hf_embeddings)

llm = OllamaLLM(model="llama3")

template = """
You are a helpful assistant. Use the provided context to answer the question. 
If the context is not sufficient, say "I don't know" and do not attempt to fabricate an answer.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    retriever=vectorstore.as_retriever(),
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

question = input("Введите ваш вопрос: ")

result = rag_chain.invoke({"query": question})

print("\nОтвет:")
print(result["result"])

print("\nИспользованный контекст:")
for doc in result["source_documents"]:
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(doc.page_content)
    print("-" * 50)