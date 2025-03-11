 🚀 Equity Research AI Bot | The Ultimate Financial News Research Tool 📊
https://github.com/user-attachments/assets/b6decabe-af13-4bbe-80f3-482ace87b382

## **📖 Overview**
Equity Research AI Bot is a **financial news research assistant** powered by **OpenAI & LangChain**. This tool allows users to:
- Fetch and analyze financial news articles 📑
- Extract insights from latest stock market trends 📈
- Answer user queries based on real-time data ⚡

This project is ideal for **investors, analysts, and finance professionals** who want **instant insights** from multiple financial sources.

---

## **✨ Features**
 **Fetch Financial News** – Input news article URLs and process real-time data.  
 **AI-Powered Research** – Uses OpenAI & FAISS to retrieve relevant information.  
 **Financial Market Analysis** – Ask questions about stocks, economy, and trends.  
 **Custom Embeddings** – Converts articles into vectorized knowledge for accurate responses.  
 **User-Friendly Interface** – Clean, structured, and interactive UI built with **Streamlit**.  
 **Personalized Insights** – Get answers tailored to your finance-related questions.  
 **Beautiful UI Enhancements** – Dark mode, aligned elements, improved typography.  

---

## **🛠️ Technologies Used**
- **[Python](https://www.python.org/)** – Core language.
- **[Streamlit](https://streamlit.io/)** – UI framework.
- **[LangChain](https://python.langchain.com/)** – AI-powered document processing.
- **[OpenAI API](https://openai.com/)** – Embeddings and response generation.
- **[FAISS](https://github.com/facebookresearch/faiss)** – Vector database for fast search.
- **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)** – Web scraping (if needed).
- **[dotenv](https://pypi.org/project/python-dotenv/)** – API key management.

---

## **🔑 Key LangChain Components Used**
This project leverages **LangChain** to process and analyze financial news articles. Below are the key components:

1. **`UnstructuredURLLoader`** – Fetches and loads unstructured text from provided URLs.
2. **`RecursiveCharacterTextSplitter`** – Splits large documents into smaller, structured chunks for better processing.
3. **`OpenAIEmbeddings`** – Converts textual content into vector embeddings for similarity search.
4. **`FAISS`** – Efficiently searches through embedded vectors for relevant financial insights.
5. **`RetrievalQAWithSourcesChain`** – A retrieval-based Q&A chain that fetches relevant context before generating responses.
6. **`OpenAI` (LLM)** – The language model used to analyze financial data and generate insightful responses.

---

## **📌 LLM Method Used**
This project uses the **Stuff Method** for processing retrieved data.  
### **📜 What is the Stuff Method?**
- The **Stuff Method** takes **all retrieved documents** and **feeds them into the LLM at once**.  
- It is efficient for **small-sized contexts** but may not work well for **large document sets** due to token limitations.

💡 **Why use the Stuff Method?**  
- Since **financial news articles are relatively short**, the **Stuff Method** ensures **maximum context utilization**.  
- This allows the model to provide **detailed, accurate responses** based on the entire article.


