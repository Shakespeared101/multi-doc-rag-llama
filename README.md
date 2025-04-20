##**🧠 multi-doc-rag-llama**
Knowledge-aware, multi-document Retrieval-Augmented Generation with Llama3 & LlamaIndex.
A graph-driven approach to consolidate hierarchical topic structures across multiple sources—without duplication#.

#**🔍 Overview**
multi-doc-rag-llama is a Retrieval-Augmented Generation (RAG) system built using open-source LLMs (e.g., Llama3), LlamaIndex, and a suite of supporting modules to:

- Handle multiple documents across various formats (TXT, PDF, DOCX, etc.)
- Model document content as interconnected hierarchical trees
- Build a knowledge graph linking topics and subtopics across documents
- Enable intelligent, non-duplicated retrieval of relevant information
- Provide rich output including relevant tables, images, and structured data

Think of it as ChatGPT meets Notion meets Knowledge Graph—on open weights.

#**📁 Project Structure**
```bash
multi-doc-rag-llama/
│
├── app.py                  # Streamlit app (UI)
├── main.py                 # Entry point for processing and querying
├── doc_loader.py           # Multi-format document loader
├── rag_index.py            # RAG setup with vector and graph-based indexes
├── knowledge_graph.py      # Builds interlinked topic-subtopic graphs
├── expanded_retriever.py   # Retrieval logic (including graph awareness)
├── query_engine.py         # Query engine with deduplication + consolidation
├── .streamlit/             # Streamlit configs
├── *.txt                   # Sample documents
├── requirements.txt        # Dependencies
└── .gitattributes          # LFS-tracked file config
```

#**🧩 Core Concepts
📚 Document Structure**
Each document is parsed into a topic → subtopic tree structure.

```
Document A:
└── Topic 1
    ├── Subtopic 1.1
    └── Subtopic 1.2

Document B:
└── Topic 2
    ├── Subtopic 2.1
    └── Topic 1 (linked from Doc A)
```

- Topics can link across documents
- Subtopics can be related to other topics or subtopics

#**🌐 Knowledge Graph**
The system builds a graph of trees, capturing:

- Cross-document topic relationships
- Relevance between topics and subtopics
- Nodes: Topics/Subtopics
- Edges: Contextual & semantic relationships

#**🧠 Query Goals**
When a user queries a topic:

- All related content (topics + subtopics) is retrieved
- Content is appended (not summarized) into a unified view
- No duplication: Repeated sections are merged intelligently
- Tables, images, and other content are included if relevant

**🚀 Getting Started
🔧 Installation**
```bash
git clone https://github.com/Shakespeared101/multi-doc-rag-llama.git
cd multi-doc-rag-llama

# Install dependencies
pip install -r requirements.txt
```

Optionally set up llama-cpp, HuggingFace Transformers, or LangChain backends depending on the LLM used.

**▶️ Run the App**
```bash
streamlit run app.py
```

**Or run via CLI:**
```bash
python main.py
```

#**📦 Features**
- ✅ Multi-format document ingestion
- ✅ Topic/subtopic parsing with hierarchy
- ✅ Cross-doc linking & knowledge graph building
- ✅ Smart deduplicated RAG retrieval
- ✅ Output includes full text, tables, and visuals
- ✅ Powered by Llama3 or compatible LLMs
- ✅ Lightweight + fully open source

#**🧠 Use Cases**
- Academic research consolidation
- Enterprise knowledge systems
- Legal document understanding
- Smart FAQs / Helpdesk systems
- Anything that needs structured + scalable multi-doc reasoning

**📸 Sample Output (Coming Soon)**
Visual examples of queries, knowledge graph snapshots, and RAG outputs.

**📜 License**
MIT License. Do what you want, but don’t blame us if your model hallucinates a recipe for plutonium.

**🤝 Contributing**
PRs welcome. If you're passionate about open knowledge, LLMs, and making machines actually useful—jump in!

**✉️ Contact**
Made with chaos and caffeine by @Shakespeared101
