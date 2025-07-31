#!/usr/bin/env python3
"""
PDF RAG System - Enhanced with Mistral LLM and Chart Generation
Load any PDF and query it with natural language using Ollama + Mistral
Installation:
pip install chromadb sentence-transformers PyPDF2 numpy faiss-cpu requests matplotlib seaborn pandas
For Ollama:
1. Install Ollama from https://ollama.ai
2. Run: ollama pull mistral
Usage:
python pdf_rag_system.py
Features:
- Load any PDF document
- Custom embeddings using all-MiniLM-L6-v2
- ChromaDB for storage, FAISS for retrieval
- Mistral LLM via Ollama for intelligent answers
- Automatic chart generation (pie, bar, line charts only)
- Enhanced interactive Q&A mode with instant chart display
"""
import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
import requests
import subprocess
import platform
warnings.filterwarnings("ignore")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'chromadb': 'chromadb',
        'sentence_transformers': 'sentence-transformers',
        'PyPDF2': 'PyPDF2',
        'numpy': 'numpy',
        'faiss': 'faiss-cpu',
        'requests': 'requests',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("Missing required packages!")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nAlso make sure Ollama is installed and Mistral model is pulled:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Run: ollama pull mistral")
        print("\nThen run this script again.")
        sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    import numpy as np
    import faiss
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    check_dependencies()

class OllamaMistralLLM:
    """Mistral LLM interface via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and Mistral is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama is not running")
            
            models = response.json().get('models', [])
            model_names = [model['name'].split(':')[0] for model in models]
            
            if 'mistral' not in model_names:
                print("Warning: Mistral model not found in Ollama!")
                print("Please run: ollama pull mistral")
                print("Available models:", model_names)
                sys.exit(1)
            
            print("Connected to Ollama - Mistral model available")
            
        except requests.exceptions.RequestException as e:
            print(f"Error: Cannot connect to Ollama at {self.base_url}")
            print("Make sure Ollama is running. Install from: https://ollama.ai")
            print("Then run: ollama pull mistral")
            sys.exit(1)
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Mistral via Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["Human:", "Assistant:"]
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: Failed to get response from Mistral (Status: {response.status_code})"
                
        except requests.exceptions.RequestException as e:
            return f"Error: Connection failed to Ollama - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_chart_instruction(self, query: str, context: str) -> Dict[str, Any]:
        """Generate chart instruction based on query and context"""
        chart_prompt = f"""Analyze the following context and determine if it contains numeric data that can be visualized in a chart.

CONTEXT: {context[:2000]}

QUERY: {query}

Your task:
1. Determine if the context contains numeric data suitable for visualization
2. If yes, identify the best chart type (pie, bar, or line)
3. Extract the data points and labels
4. Provide a clear chart title

Respond in this exact JSON format:
{{
"has_data": true/false,
"chart_type": "pie" or "bar" or "line",
"title": "Chart title",
"data": {{"labels": ["label1", "label2"], "values": [value1, value2]}},
"description": "Brief description of what the chart shows"
}}

If no suitable numeric data is found, set has_data to false.
"""
        try:
            response = self.generate_response(chart_prompt, max_tokens=500)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                chart_data = json.loads(json_match.group())
                return chart_data
            else:
                return {"has_data": False}
        except:
            return {"has_data": False}

class ChartGenerator:
    """Generates pie, bar, and line charts only"""
    
    def __init__(self, output_dir: str = "./charts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
    
    def open_chart(self, filepath: str):
        """Open chart file automatically based on OS"""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(filepath)
            elif system == "Darwin":
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
        except Exception as e:
            print(f"Could not auto-open chart: {e}")
            print(f"Chart saved at: {filepath}")
    
    def generate_chart_from_llm_data(self, chart_data: Dict[str, Any]) -> Optional[str]:
        """Generate chart based on LLM-provided data"""
        if not chart_data.get("has_data", False):
            return None
        
        try:
            chart_type = chart_data.get("chart_type", "bar")
            title = chart_data.get("title", "Chart")
            data = chart_data.get("data", {})
            labels = data.get("labels", [])
            values = data.get("values", [])
            
            if not labels or not values or len(labels) != len(values):
                return None
            
            try:
                values = [float(v) for v in values]
            except:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if chart_type == "pie":
                filepath = self._create_pie_chart(labels, values, title, ax)
            elif chart_type == "line":
                filepath = self._create_line_chart(labels, values, title, ax)
            else:
                filepath = self._create_bar_chart(labels, values, title, ax)
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{chart_type}_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.open_chart(filepath)
            return filepath
            
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None
    
    def _create_pie_chart(self, labels: List[str], values: List[float], title: str, ax):
        """Create pie chart"""
        values = [abs(v) for v in values]
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(title, fontsize=16, fontweight='bold')
        return ax
    
    def _create_bar_chart(self, labels: List[str], values: List[float], title: str, ax):
        """Create bar chart"""
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, values, color=sns.color_palette("husl", len(labels)))
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        return ax
    
    def _create_line_chart(self, labels: List[str], values: List[float], title: str, ax):
        """Create line chart"""
        ax.plot(range(len(labels)), values, marker='o', linewidth=3, markersize=8)
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        return ax

class EmbeddingGenerator:
    """Handles embedding generation using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0]

class PDFProcessor:
    """Handles PDF loading and text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF (.pdf extension)")
        
        try:
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += f"\n\n--- Page {page_num} ---\n\n"
                            text_content += page_text
                    except Exception as e:
                        continue
            
            if not text_content.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            return text_content
            
        except Exception as e:
            raise

class TextChunker:
    """Handles text chunking for optimal vector storage"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, source_file: str) -> List[Dict]:
        """Split text into overlapping chunks with metadata"""
        cleaned_text = self._clean_text(text)
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(cleaned_text):
            end = start + self.chunk_size
            
            if end < len(cleaned_text):
                sentence_break = cleaned_text.rfind('.', start, end)
                if sentence_break > start + self.chunk_size - 200:
                    end = sentence_break + 1
                else:
                    para_break = cleaned_text.rfind('\n\n', start, end)
                    if para_break > start + self.chunk_size - 300:
                        end = para_break + 2
            
            chunk_text = cleaned_text[start:end].strip()
            
            if chunk_text:
                page_info = self._extract_page_info(chunk_text)
                chunk = {
                    "chunk_id": f"chunk_{chunk_id:04d}",
                    "text": chunk_text,
                    "source_file": source_file,
                    "chunk_index": chunk_id,
                    "start_char": start,
                    "end_char": end,
                    "length": len(chunk_text),
                    "page_info": page_info,
                    "created_at": datetime.now().isoformat()
                }
                chunks.append(chunk)
                chunk_id += 1
            
            start = end - self.chunk_overlap
            if start >= len(cleaned_text):
                break
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'--- Page \d+ ---', '\n\n', text)
        return text.strip()
    
    def _extract_page_info(self, chunk_text: str) -> Optional[int]:
        """Extract page number from chunk if available"""
        page_match = re.search(r'--- Page (\d+) ---', chunk_text)
        if page_match:
            return int(page_match.group(1))
        return None

class ChromaDBVectorStore:
    """ChromaDB-based vector storage for PDF chunks"""
    
    def __init__(self, persist_directory: str = "./pdf_chroma_data", embedding_generator: EmbeddingGenerator = None):
        self.persist_directory = persist_directory
        self.collection_name = "pdf_documents"
        self.embedding_generator = embedding_generator
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def create_collection(self, collection_name: str = None):
        """Create or get collection for storing chunks"""
        if collection_name:
            self.collection_name = collection_name
        
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document chunks for RAG system with custom embeddings"}
            )
    
    def add_chunks(self, chunks: List[Dict]):
        """Add text chunks to ChromaDB with custom embeddings"""
        if not chunks:
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk["text"])
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "length": chunk["length"],
                "page_info": chunk["page_info"] if chunk["page_info"] else -1,
                "created_at": chunk["created_at"]
            }
            metadatas.append(metadata)
            ids.append(chunk["chunk_id"])
        
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_generator.generate_embeddings(documents)
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
        except Exception as e:
            try:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings.tolist()
                )
            except Exception as e2:
                raise
    
    def get_all_data(self) -> Dict:
        """Get all data from ChromaDB for FAISS indexing"""
        try:
            results = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
            return results
        except Exception as e:
            return {'documents': [], 'metadatas': [], 'embeddings': []}
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory
            }
    
    def reset_collection(self):
        """Reset the collection (delete all data)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.create_collection()
        except Exception as e:
            pass

class FAISSRetriever:
    """FAISS-based retrieval system for semantic similarity search"""
    
    def __init__(self, embedding_dim: int, persist_directory: str = "./pdf_chroma_data"):
        self.embedding_dim = embedding_dim
        self.persist_directory = persist_directory
        self.faiss_index_path = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_path = os.path.join(persist_directory, "faiss_metadata.pkl")
        self.index = None
        self.metadata = []
        self.documents = []
        self._load_index()
    
    def build_index(self, embeddings: np.ndarray, documents: List[str], metadatas: List[Dict]):
        """Build FAISS index from embeddings"""
        if len(embeddings) == 0:
            return
        
        print(f"Building FAISS index with {len(embeddings)} vectors...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.documents = documents
        self.metadata = metadatas
        self._save_index()
        print(f"FAISS index built successfully with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents using FAISS"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    "rank": i + 1,
                    "similarity_score": float(similarity),
                    "chunk_text": self.documents[idx],
                    "chunk_id": self.metadata[idx]["chunk_id"],
                    "source_file": self.metadata[idx]["source_file"],
                    "page_info": self.metadata[idx].get("page_info", -1),
                    "chunk_index": self.metadata[idx]["chunk_index"],
                    "length": self.metadata[idx]["length"]
                }
                results.append(result)
        
        return results
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            if self.index is not None:
                faiss.write_index(self.index, self.faiss_index_path)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
        except Exception as e:
            print(f"Warning: Could not save FAISS index: {e}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.faiss_index_path)
                
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Could not load existing FAISS index: {e}")
            self.index = None
            self.metadata = []
            self.documents = []

class PDFRAGSystem:
    """Main PDF RAG system with Mistral LLM and simplified chart generation"""
    
    def __init__(self, persist_directory: str = "./pdf_chroma_data"):
        self.embedding_generator = EmbeddingGenerator("all-MiniLM-L6-v2")
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.vector_store = ChromaDBVectorStore(persist_directory, self.embedding_generator)
        self.faiss_retriever = FAISSRetriever(self.embedding_generator.embedding_dim, persist_directory)
        self.llm = OllamaMistralLLM()
        self.chart_generator = ChartGenerator()
        
        self.vector_store.create_collection()
        self.loaded_documents = {}
        self._sync_faiss_with_chromadb()
    
    def _sync_faiss_with_chromadb(self):
        """Ensure FAISS index is synced with ChromaDB data"""
        chromadb_data = self.vector_store.get_all_data()
        
        if (len(chromadb_data.get('documents', [])) > 0 and 
            (self.faiss_retriever.index is None or self.faiss_retriever.index.ntotal == 0)):
            print("Syncing FAISS index with existing ChromaDB data...")
            embeddings = np.array(chromadb_data['embeddings'], dtype='float32')
            documents = chromadb_data['documents']
            metadatas = chromadb_data['metadatas']
            self.faiss_retriever.build_index(embeddings, documents, metadatas)
    
    def load_pdf(self, pdf_path: str, force_reload: bool = False):
        """Load and process a PDF document"""
        pdf_name = os.path.basename(pdf_path)
        
        if pdf_name in self.loaded_documents and not force_reload:
            return
        
        try:
            pdf_text = self.pdf_processor.load_pdf(pdf_path)
            chunks = self.text_chunker.chunk_text(pdf_text, pdf_name)
            self.vector_store.add_chunks(chunks)
            self._rebuild_faiss_index()
            
            self.loaded_documents[pdf_name] = {
                "path": pdf_path,
                "chunks_count": len(chunks),
                "text_length": len(pdf_text),
                "loaded_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index with all ChromaDB data"""
        chromadb_data = self.vector_store.get_all_data()
        if len(chromadb_data.get('documents', [])) > 0:
            embeddings = np.array(chromadb_data['embeddings'], dtype='float32')
            documents = chromadb_data['documents']
            metadatas = chromadb_data['metadatas']
            self.faiss_retriever.build_index(embeddings, documents, metadatas)
    
    def _create_llm_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """Create a comprehensive prompt for Mistral LLM"""
        context_text = ""
        for i, chunk in enumerate(context_chunks[:3], 1):
            page_info = f" (Page {chunk['page_info']})" if chunk['page_info'] > 0 else ""
            context_text += f"\n--- Source {i}{page_info} ---\n"
            context_text += chunk['chunk_text']
            context_text += "\n"
        
        prompt = f"""You are an expert AI assistant analyzing PDF document content. Your task is to provide accurate, comprehensive answers based on the provided context.

CONTEXT FROM PDF:
{context_text}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer the question directly and comprehensively using the provided context
2. Write your response as a single, well-structured paragraph
3. Focus on the most relevant information from the context
4. Be concise but thorough
5. Use a professional yet conversational tone
6. Do not mention sources or chunks in your response

ANSWER:"""
        return prompt
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Enhanced query with Mistral LLM and automatic chart generation"""
        query_embedding = self.embedding_generator.generate_single_embedding(question)
        search_results = self.faiss_retriever.search(query_embedding, top_k)
        
        if not search_results:
            return {
                "query": question,
                "timestamp": datetime.now().isoformat(),
                "found_results": False,
                "llm_answer": "I couldn't find any relevant information about your question in the PDF document. Please try rephrasing your question or ask about different topics that might be covered in the document.",
                "confidence": 0.0,
                "chart_generated": None
            }
        
        llm_prompt = self._create_llm_prompt(question, search_results)
        llm_answer = self.llm.generate_response(llm_prompt, max_tokens=1500)
        
        combined_context = " ".join([chunk["chunk_text"] for chunk in search_results])
        chart_generated = None
        
        chart_data = self.llm.generate_chart_instruction(question, combined_context)
        if chart_data.get("has_data", False):
            print("Generating chart from extracted data...")
            chart_path = self.chart_generator.generate_chart_from_llm_data(chart_data)
            if chart_path:
                chart_generated = {
                    "path": chart_path,
                    "type": chart_data.get("chart_type", "bar"),
                    "title": chart_data.get("title", "Chart"),
                    "description": chart_data.get("description", "")
                }
                print(f"Chart generated and opened: {os.path.basename(chart_path)}")
        
        avg_similarity = np.mean([result["similarity_score"] for result in search_results])
        confidence = min(avg_similarity * 100, 95)
        
        response = {
            "query": question,
            "timestamp": datetime.now().isoformat(),
            "found_results": True,
            "llm_answer": llm_answer,
            "confidence": round(confidence, 1),
            "chart_generated": chart_generated
        }
        
        return response
    
    def show_system_stats(self):
        """Display simplified system statistics"""
        collection_info = self.vector_store.get_collection_info()
        faiss_count = self.faiss_retriever.index.ntotal if self.faiss_retriever.index else 0
        
        print(f"\n{'='*80}")
        print("PDF RAG SYSTEM STATISTICS")
        print("="*80)
        print(f"LLM Model: Mistral (via Ollama)")
        print(f"Embedding Model: {self.embedding_generator.model_name}")
        print(f"Chart Types: Pie Charts, Bar Charts, Line Charts")
        print(f"ChromaDB Chunks: {collection_info['document_count']}")
        print(f"FAISS Index Size: {faiss_count}")
        print(f"Loaded Documents: {len(self.loaded_documents)}")
        print(f"Charts Directory: {self.chart_generator.output_dir}")
        
        if self.loaded_documents:
            print(f"\nLOADED DOCUMENTS:")
            for doc_name, info in self.loaded_documents.items():
                print(f" {doc_name} - {info['chunks_count']} chunks")
    
    def interactive_mode(self):
        """Simplified interactive Q&A mode with automatic chart opening"""
        print(f"\n{'='*80}")
        print("PDF RAG SYSTEM - INTERACTIVE MODE")
        print("Powered by Mistral LLM + Auto Chart Generation")
        print("="*80)
        
        if not self.loaded_documents:
            print("\nNo PDF documents loaded!")
            print("Load a PDF first using: system.load_pdf('path/to/your/file.pdf')")
            return
        
        self.show_system_stats()
        
        print(f"\nAsk questions about your PDF content:")
        print("Examples:")
        print(" • 'What is the main topic of this document?'")
        print(" • 'Summarize the key findings'")
        print(" • 'Show me the statistics'")
        print(" • 'What are the numerical trends?'")
        print(" • 'Create a visualization of the data'")
        print(f"\nCharts will auto-open when numeric data is found!")
        print(f"Type 'quit' to exit")
        print("-" * 80)
        
        while True:
            try:
                user_question = input("\nYour Question: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_question:
                    continue
                
                print("\nAnalyzing PDF content...")
                response = self.query(user_question)
                
                if response["found_results"]:
                    print(f"\nANSWER (Confidence: {response['confidence']}%):")
                    print("="*60)
                    print(response["llm_answer"])
                    
                    if response["chart_generated"]:
                        chart_info = response["chart_generated"]
                        print(f"\nCHART GENERATED & OPENED:")
                        print(f" Type: {chart_info['type'].title()} Chart")
                        print(f" Title: {chart_info['title']}")
                        print(f" File: {os.path.basename(chart_info['path'])}")
                        if chart_info['description']:
                            print(f" Description: {chart_info['description']}")
                else:
                    print(f"\n{response['llm_answer']}")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
        
        print("\nInteractive mode ended. Thank you for using the PDF RAG System!")

def main():
    """Main function to run the enhanced PDF RAG system"""
    print("="*80)
    print("ENHANCED PDF RAG SYSTEM")
    print("Mistral LLM + Auto Chart Generation (Pie/Bar/Line)")
    print("="*80)
    print("Load any PDF and query it with intelligent AI responses!")
    
    try:
        check_dependencies()
        
        print(f"\n{'='*60}")
        print("INITIALIZING SYSTEM")
        print("="*60)
        
        rag_system = PDFRAGSystem()
        
        pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
        
        if pdf_files:
            print(f"\nFound PDF files in current directory:")
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f" {i}. {pdf_file}")
            
            while True:
                try:
                    choice = input(f"\nSelect PDF to load (1-{len(pdf_files)}) or enter path: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(pdf_files):
                        selected_pdf = pdf_files[int(choice) - 1]
                        break
                    elif os.path.exists(choice) and choice.lower().endswith('.pdf'):
                        selected_pdf = choice
                        break
                    else:
                        print("Invalid selection or file not found. Try again.")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    return
        else:
            selected_pdf = input("Enter path to PDF file: ").strip()
            if not os.path.exists(selected_pdf):
                print(f"File not found: {selected_pdf}")
                return
        
        print(f"\nLoading PDF: {selected_pdf}")
        rag_system.load_pdf(selected_pdf)
        
        rag_system.interactive_mode()
        
        print(f"\nPDF RAG SYSTEM SESSION COMPLETED!")
        print("="*80)
        
    except KeyboardInterrupt:
        print(f"\n\nSystem interrupted by user")
    except Exception as e:
        print(f"\nSystem error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()