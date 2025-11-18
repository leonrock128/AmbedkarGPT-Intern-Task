from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os


def build_vector_db():
    print(" Loading speech.txt ...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "speech.txt")

    loader = TextLoader(file_path)
    documents = loader.load()

    print(" Splitting text ...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print("CHUNKS CREATED:", len(chunks))

    print(" Creating embeddings ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(" Saving to ChromaDB ...")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=os.path.join(base_dir, "db")
    )
    vectordb.persist()

    print(" Vector DB created!")
    return vectordb


def load_vector_db():
    print(" Loading existing ChromaDB ...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=os.path.join(base_dir, "db"),
        embedding_function=embeddings
    )


def start_qa(vectordb):
    print(" Loading Mistral (Ollama) ...")
    llm = Ollama(model="phi3")


    # retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    print("\n AmbedkarGPT Ready! Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower().strip() == "exit":
            break

        # FIXED: use .invoke() instead of get_relevant_documents()
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        print("\n Answer:")
        print(response)

        print("\n Context Used:\n")
        for i, doc in enumerate(docs):
            print(f"[{i+1}] {doc.page_content[:300]}...\n")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "db")
    
    if not os.path.exists(db_path):
        vectordb = build_vector_db()
    else:
        vectordb = load_vector_db()

    start_qa(vectordb)