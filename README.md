# Medical Chatbot using LLM

A sophisticated medical chatbot application powered by Large Language Models (LLM) that provides intelligent responses to medical queries. This project leverages natural language processing and retrieval-augmented generation (RAG) to deliver accurate and contextual medical information.

## 🌟 Features

- **Intelligent Medical Q&A**: Get answers to medical questions using advanced LLM technology
- **Context-Aware Responses**: Utilizes vector database for relevant information retrieval
- **User-Friendly Interface**: Clean and intuitive chat interface
- **Document Processing**: Processes medical documents and knowledge bases
- **Real-time Interaction**: Fast response times with efficient model inference
- **Conversation History**: Maintains context throughout the conversation

## 🛠️ Technology Stack

- **Language Model**: LLM (Llama 2 / GPT / Other)
- **Framework**: LangChain
- **Vector Database**: Pinecone / FAISS / Chroma
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript / Streamlit / Chainlit
- **Embeddings**: HuggingFace Sentence Transformers
- **Libraries**: 
  - LangChain
  - PyTorch
  - Transformers
  - FAISS / Pinecone
  - Flask

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Minimum 8GB RAM (16GB recommended for optimal performance)
- GPU (optional, but recommended for faster inference)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/deep-khimani/Medical-Chatbot-using-LLM.git
   cd Medical-Chatbot-using-LLM
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory and add your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENV=your_pinecone_environment
   HUGGINGFACE_API_TOKEN=your_huggingface_token
   ```

5. **Download the LLM model**
   
   Follow the instructions to download your chosen LLM model or configure API access.

## 📊 Data Preparation

1. Place your medical documents (PDFs, text files) in the `data/` directory
2. Run the data ingestion script:
   ```bash
   python ingest.py
   ```
3. This will process the documents and create embeddings in your vector database

## 💻 Usage

### Running the Application

**Option 1: Flask Application**
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

**Option 2: Streamlit Application**
```bash
streamlit run streamlit_app.py
```

**Option 3: Chainlit Application**
```bash
chainlit run chainlit_app.py
```

### Example Queries

- "What are the symptoms of diabetes?"
- "How can I manage high blood pressure?"
- "What is the recommended dosage for common pain relievers?"
- "Explain the difference between Type 1 and Type 2 diabetes"

## 📁 Project Structure

```
Medical-Chatbot-using-LLM/
│
├── data/                      # Medical documents and datasets
├── models/                    # Downloaded LLM models
├── vectorstore/               # Vector database storage
├── src/
│   ├── helper.py             # Helper functions
│   ├── prompt.py             # Prompt templates
│   └── embedding.py          # Embedding utilities
├── app.py                    # Main Flask application
├── ingest.py                 # Data ingestion script
├── model.py                  # Model configuration
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
└── README.md                 # Project documentation
```

## ⚙️ Configuration

### Model Configuration

Edit `model.py` to configure your LLM settings:
- Model type and version
- Temperature and other generation parameters
- Context window size
- Maximum token limits

### Vector Database Configuration

Configure your vector database settings in the respective configuration files:
- Index name
- Dimension size
- Similarity metrics
- Number of results to retrieve

## 🔒 Important Notes

- **Medical Disclaimer**: This chatbot is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
- **Data Privacy**: Ensure compliance with HIPAA and other relevant healthcare data regulations
- **Model Limitations**: LLMs may occasionally generate incorrect or outdated information
- **Use Responsibly**: Always verify critical medical information with qualified healthcare professionals

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes and commit (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Deep Khimani** - [GitHub Profile](https://github.com/deep-khimani)

## 🙏 Acknowledgments

- LangChain for the powerful framework
- HuggingFace for embeddings and transformers
- The open-source LLM community
- Medical datasets and knowledge bases used for training

## 📧 Contact

For questions, suggestions, or collaboration:
- GitHub: [@deep-khimani](https://github.com/deep-khimani)
- Open an issue in this repository

## 🐛 Known Issues

- Large documents may take time to process
- Response time may vary based on hardware capabilities
- Some complex medical queries may require multiple interactions

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output capabilities
- [ ] Integration with medical databases and APIs
- [ ] Enhanced conversation memory
- [ ] Mobile application
- [ ] Fine-tuning on specialized medical datasets
- [ ] User authentication and session management

---

**⚠️ Disclaimer**: This is an educational project. Always consult with qualified healthcare professionals for medical advice.
