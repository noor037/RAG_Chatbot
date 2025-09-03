# 📖 RAG Chatbot with Online Articles

An **AI-powered Retrieval-Augmented Generation (RAG) chatbot** that allows you to **add any online article** and interact with it through natural conversations.
Built with **LangChain, Groq LLMs, HuggingFace embeddings, ChromaDB, and Streamlit**, this project enables users to **ask questions** and get **context-aware answers** based on the provided articles.

---

## 🚀 Features

* 🔗 **Add Articles Dynamically** – Provide any URL, and the bot fetches + processes the content.
* 🧠 **Retrieval-Augmented Generation** – Combines **vector search** with **Groq’s LLaMA-3.3-70B model** for accurate responses.
* ⚡ **Fast & Scalable** – Uses **Groq API** for blazing fast inference.
* 📚 **Persistent Knowledge Base** – Articles are stored in **ChromaDB** for future queries.
* 💬 **Interactive Chat Interface** – Built on **Streamlit chat UI** for a smooth experience.
* 🔑 **Secure API Key Input** – Enter your **Groq API Key** safely in the sidebar.

---

## 🛠️ Tech Stack

* **[Streamlit](https://streamlit.io/)** – UI framework for interactive apps
* **[LangChain](https://www.langchain.com/)** – Orchestration of RAG pipeline
* **[Groq LLMs](https://groq.com/)** – High-performance LLaMA model inference
* **[HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** – Embedding model
* **[ChromaDB](https://www.trychroma.com/)** – Vector database for storing article embeddings

---

## 📂 Project Structure

```
📦 RAG-Chatbot
 ┣ 📂 chroma_db        # Persisted vector database
 ┣ 📜 stream.py           # Main Streamlit app
 ┣ 📜 requirements.txt # Dependencies
 ┗ 📜 README.md        # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/noor037/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run stream.py
   ```

---

## 🔑 API Key Setup

* Get your **Groq API Key** from [Groq Console](https://console.groq.com/).
* Enter it in the **Streamlit sidebar** under **Settings**.

---

## 🎯 Usage

1. Open the Streamlit app in your browser.
2. Add a new article URL.
3. Ask questions in the chatbox.
4. Get **context-aware answers** directly from the articles.

---

## 📸 Demo Preview

(👉 Add screenshot/GIF of your chatbot UI here)

---

## 🤝 Contributing

Contributions are welcome!
If you’d like to improve the app (UI/UX, new models, multi-doc support), feel free to **fork the repo** and submit a PR.

---

## 📜 License

This project is licensed under the **MIT License**.

---

✨ Built with ❤️ using **LangChain, Groq, and Streamlit**
Noor Alam 
(AI/ML Engineer)
