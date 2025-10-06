# src/rag/rag_pipeline.py
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class RAGPipeline:
    def __init__(self, knowledge_dir="data/knowledge_base"):
        """
        Initialize RAG system:
        - HuggingFace embeddings for retrieval
        - Google Gemini via google-generativeai SDK for text generation
        """
        self.knowledge_dir = knowledge_dir

        # 1️⃣ Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2️⃣ Vectorstore for knowledge retrieval
        self.vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 3️⃣ Initialize Google Generative AI
        genai.configure(api_key="")  # Set your key here or via environment variable

    def generate_insight(self, detected_objects):
        """
        Takes YOLO detections and generates textual insights.
        """
        if not detected_objects:
            return "No objects detected in the image/video."

        # Prepare query text from YOLO detections
        object_list = [obj["object"] for obj in detected_objects]
        query = f"Objects detected: {', '.join(object_list)}. Provide a concise context-aware analysis and suggestions."

        # Retrieve relevant knowledge chunks
        relevant_docs = self.retriever.get_relevant_documents(query)
        context_text = "\n".join([doc.page_content for doc in relevant_docs])

        # Generate insight using Google Gemini
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Context:\n{context_text}\n\nBased on the detected objects, generate a concise insight:\n{query}"
        response = model.generate_content(  
            contents=[{"text": prompt}],
            
        )

        insight_text = response.text.strip()
        return insight_text

