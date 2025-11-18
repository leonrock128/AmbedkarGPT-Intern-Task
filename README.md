
# AmbedkarGPT – Local RAG Question Answering System using Ollama + LangChain

AmbedkarGPT is a locally running **Retrieval-Augmented Generation (RAG)** application that answers questions from **Dr. B. R. Ambedkar’s speeches**, completely offline.

It uses:

- **LangChain 1.x**
- **ChromaDB vector database**
- **HuggingFace Embeddings**
- **Ollama (LLaMA 3.2 1B model – lightweight)**

This project runs fully on your local machine without internet.

---

## 🚀 Features

- Load text data from `speech.txt`
- Split text into semantic chunks
- Convert chunks into embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Retrieve relevant context for any user query
- Answer questions using a small local LLM (Ollama)
- Works on low-RAM laptops (using `llama3.2:1b`)

---

## 📁 Project Structure

```

AmbedkarGPT/
│
├── main.py                  # Main RAG application
├── speech.txt               # Text source (Ambedkar content)
├── db/                      # Vector database (auto-created)
├── README.md                # Project documentation
└── .gitignore               # Ignore venv/db/cache files

````

---

## 🧠 Requirements

Install dependencies:

```bash
pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers
````

Also install a lightweight model in Ollama:

```bash
ollama pull llama3.2:1b
```

---

## 🔧 How to Run

1. Start Ollama in another terminal:

```bash
ollama serve
```

2. Run the application:

```bash
python main.py
```

3. Ask questions:

```
What does Ambedkar say about Shastras?
```

4. Exit:

```
exit
```

---

## 🛠 How It Works (RAG Pipeline)

1. **Load Data**
   The content from `speech.txt` is loaded using LangChain's TextLoader.

2. **Chunk the Text**
   Using CharacterTextSplitter, the text is split into 500-character chunks.

3. **Create Embeddings**
   Using HuggingFace:
   `sentence-transformers/all-MiniLM-L6-v2`

4. **Store in Vector DB**
   ChromaDB stores all embeddings and metadata.

5. **Retrieve Relevant Chunks**
   For every user question, top 2 matching chunks are retrieved.

6. **Generate Answer**
   LLaMA 3.2 (1B) model uses retrieved context to produce accurate answers.

---

## 🧪 Example Output

```
You: What is Ambedkar saying about Shastras?

Answer:
Dr. Ambedkar argues that the Shastras uphold the caste system and prevent social equality...

Context Used:
[1] "... text from speech.txt ... "
[2] "... text from speech.txt ... "
```

---

## 💻 System Requirements

* Windows / Linux / Mac
* Python 3.10+
* 2GB RAM minimum
* Ollama installed

Small model (`llama3.2:1b`) ensures compatibility with low-end machines.

---

## 🧾 .gitignore (recommended)

```
venv/
db/
__pycache__/
*.cache
.vscode/
.idea/
```

---

## 🤝 Contributing

Pull requests are welcome!
To contribute:

```bash
git checkout -b my-feature
git commit -m "Add feature"
git push origin my-feature
```

Then open a Pull Request.

---

## 📜 License

This project is open-source under the MIT License.

---

## ✨ Author

**Ravi L**
Built with ❤️ using LangChain + Ollama

```

---
