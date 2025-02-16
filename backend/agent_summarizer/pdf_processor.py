# import os
# import base64
# import io
# from pathlib import Path
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import MongoDBAtlasVectorSearch
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain_community.document_loaders import PyPDFLoader
# from pdf2image import convert_from_path
# import pytesseract
# from PIL import Image
# import fitz
# from openai import OpenAI
# import json
# from datetime import datetime
# from bson import ObjectId

# # Load environment variables
# env_path = Path(__file__).parent / ".env"
# load_dotenv(env_path)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found in .env file")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# MONGO_URI = os.getenv("MONGO_URI")

# # MongoDB Connection
# if MONGO_URI:
#     client = MongoClient(MONGO_URI)
#     db = client["rag_demo"]
#     collection = db["pdf_data"]
# else:
#     print("Warning: MONGO_URI not found in .env file")

# class MongoJSONEncoder(json.JSONEncoder):
#     def default(self, o):
#         if isinstance(o, ObjectId):
#             return str(o)
#         return super().default(o)

# class PDFProcessor:
#     def __init__(self, pdf_path=None):
#         self.pdf_path = pdf_path
#         self.vector_store = None
#         self.retriever = None
#         self.chunk_size = 1000
#         self.chunk_overlap = 200
#         self.separator = "\n\n"
#         self.processed_data = {
#             "metadata": {
#                 "processed_at": "",
#                 "file_path": "",
#                 "processing_stats": {}
#             },
#             "content": {
#                 "full_text": "",
#                 "summary": "",
#                 "qa_results": []
#             }
#         }

#     def set_pdf_path(self, pdf_path):
#         """Set or update the PDF path"""
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF file not found: {pdf_path}")
#         self.pdf_path = pdf_path

#     def extract_text_from_pdf(self):
#         """Extract text content from PDF using PyPDFLoader"""
#         if not self.pdf_path:
#             raise ValueError("PDF path not set")
        
#         loader = PyPDFLoader(self.pdf_path)
#         documents = loader.load()
#         return "\n".join([doc.page_content for doc in documents])

#     def extract_images_from_pdf(self):
#         """Extract images from PDF pages"""
#         if not self.pdf_path:
#             raise ValueError("PDF path not set")
        
#         images = convert_from_path(self.pdf_path)
#         image_paths = []
#         for i, image in enumerate(images):
#             image_path = f"page_{i+1}.jpg"
#             image.save(image_path, "JPEG")
#             image_paths.append(image_path)
#         return image_paths

#     def extract_text_from_images(self, image_paths, method="tesseract"):
#         """Perform OCR on extracted images using specified method"""
#         ocr_texts = []
        
#         if method == "tesseract":
#             for image_path in image_paths:
#                 text = pytesseract.image_to_string(Image.open(image_path))
#                 ocr_texts.append(text)
                
#         elif method == "openai":
#             client = OpenAI(api_key=OPENAI_API_KEY)
#             for image_path in image_paths:
#                 with open(image_path, "rb") as image_file:
#                     base64_image = base64.b64encode(image_file.read()).decode("utf-8")
#                     response = client.chat.completions.create(
#                         model="gpt-4-vision-preview",
#                         messages=[
#                             {"role": "system", "content": "Extract key information from this image."},
#                             {"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}]},
#                         ]
#                     )
#                     ocr_texts.append(response.choices[0].message.content)
                    
#         # Clean up temporary image files
#         for image_path in image_paths:
#             try:
#                 os.remove(image_path)
#             except OSError:
#                 pass
                
#         return "\n".join(ocr_texts)

#     def process_pdf(self, ocr_method="tesseract"):
#         """Process PDF file and extract all content with improved text handling"""
#         if not self.pdf_path:
#             raise ValueError("PDF path not set")
        
#         start_time = datetime.now()
        
#         # Extract text and images
#         pdf_text = self.extract_text_from_pdf()
#         image_paths = self.extract_images_from_pdf()
#         ocr_text = self.extract_text_from_images(image_paths, method=ocr_method)
        
#         # Preprocess and combine text
#         final_text = self._preprocess_text(pdf_text + "\n\n" + ocr_text)
        
#         # Update processed data
#         self.processed_data["metadata"].update({
#             "processed_at": datetime.now().isoformat(),
#             "file_path": self.pdf_path,
#             "processing_stats": {
#                 "processing_time": str(datetime.now() - start_time),
#                 "total_chars": len(final_text),
#                 "total_words": len(final_text.split())
#             }
#         })
#         self.processed_data["content"]["full_text"] = final_text
        
#         # Store in vector database if MongoDB is configured
#         if MONGO_URI:
#             self._store_in_vector_db(final_text)
            
#         return final_text

#     def _preprocess_text(self, text):
#         """Preprocess text for better quality"""
#         # Remove excessive whitespace
#         text = " ".join(text.split())
#         # Restore paragraph breaks
#         text = text.replace(". ", ".\n\n")
#         # Remove any non-printable characters
#         text = "".join(char for char in text if char.isprintable())
#         return text

#     def _store_in_vector_db(self, text):
#         """Store the processed text in MongoDB vector store with improved chunking"""
#         if not MONGO_URI:
#             raise ValueError("MongoDB URI not configured")
            
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             separators=[self.separator, "\n", ". ", " ", ""],
#             length_function=len
#         )
#         texts = text_splitter.create_documents([text])
        
#         # Add metadata to chunks for better context
#         for i, doc in enumerate(texts):
#             doc.metadata["chunk_id"] = i
#             doc.metadata["total_chunks"] = len(texts)
            
#         embeddings = OpenAIEmbeddings()
        
#         self.vector_store = MongoDBAtlasVectorSearch.from_documents(
#             texts, 
#             embeddings, 
#             collection=collection, 
#             index_name="vector_index"
#         )
#         self.retriever = self.vector_store.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 5}
#         )

#     def summarize_text(self, text):
#         """Summarize the extracted text using GPT-4 with improved prompting"""
#         llm = ChatOpenAI(temperature=0)
        
#         # Create a structured prompt for better summaries
#         prompt = """Please provide a comprehensive summary of the following text in JSON format with these sections:
#         {
#             "main_topics": ["List of main topics"],
#             "key_concepts": ["List of key concepts"],
#             "findings": ["List of important findings"],
#             "details": ["List of significant details"],
#             "summary": "Overall summary text"
#         }

#         Text to summarize:
#         """ + text[:4000]

#         response = llm.invoke(prompt)
#         try:
#             summary_data = json.loads(response.content)
#         except json.JSONDecodeError:
#             # If JSON parsing fails, create a simple format
#             summary_data = {
#                 "main_topics": [],
#                 "key_concepts": [],
#                 "findings": [],
#                 "details": [],
#                 "summary": response.content
#             }
#         self.processed_data["content"]["summary"] = summary_data
#         return summary_data

#     def query_document(self, query):
#         """Query the processed document using RAG with improved response formatting"""
#         if not self.retriever:
#             raise ValueError("Document not processed or vector store not initialized")
            
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=ChatOpenAI(temperature=0),
#             chain_type="stuff",
#             retriever=self.retriever,
#             return_source_documents=True
#         )
        
#         try:
#             result = qa_chain({"query": query})
#             answer = result["result"]
#             sources = result["source_documents"]
            
#             qa_result = {
#                 "question": query,
#                 "answer": answer,
#                 "sources": [
#                     {
#                         "content": doc.page_content,  # Include full content
#                         "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
#                         "total_chunks": doc.metadata.get("total_chunks", "Unknown"),
#                         "source": doc.metadata.get("source", "Unknown"),
#                         "metadata": doc.metadata
#                     }
#                     for doc in sources
#                 ]
#             }
            
#             self.processed_data["content"]["qa_results"].append(qa_result)
#             return qa_result
            
#         except Exception as e:
#             error_data = {"error": str(e), "query": query}
#             return error_data

#     def save_results(self, output_path):
#         """Save processing results to a JSON file"""
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(self.processed_data, f, ensure_ascii=False, indent=2, cls=MongoJSONEncoder)

#     def get_results(self):
#         """Get processing results as a dictionary"""
#         return json.loads(json.dumps(self.processed_data, cls=MongoJSONEncoder))

#     def cleanup(self):
#         """Clean up resources"""
#         if MONGO_URI and hasattr(self, 'client'):
#             self.client.close()

# # Example usage
# if __name__ == "__main__":
#     pdf_path = "../example_data/chap4.pdf"
#     processor = PDFProcessor(pdf_path)
    
#     try:
#         print("Processing PDF...")
#         extracted_text = processor.process_pdf(ocr_method="tesseract")
#         print("PDF processing completed.")
        
#         print("\nGenerating summary...")
#         summary = processor.summarize_text(extracted_text)
#         print("\n" + "="*50)
#         print("PDF SUMMARY")
#         print("="*50)
#         print(json.dumps(summary, indent=2))
#         print("="*50 + "\n")
        
#         if processor.retriever:
#             queries = [
#                 "What are the main topics covered in this chapter?",
#                 "What are the key concepts discussed in this chapter?",
#                 "Summarize the most important points from this chapter."
#             ]
            
#             print("QUESTION & ANSWERS")
#             print("="*50)
#             for query in queries:
#                 print(f"\nQ: {query}")
#                 result = processor.query_document(query)
#                 if result:
#                     print("\nA:", result["answer"])
#                     print("\nSources:")
#                     for source in result["sources"]:
#                         print(f"- Chunk {source['chunk_id']}/{source['total_chunks']}")
#                         print(f"  Content: {source['content'][:200]}...")
#                 print("-" * 50)
                
#     except Exception as e:
#         print(f"Error: {str(e)}")
#     finally:
#         processor.cleanup()
