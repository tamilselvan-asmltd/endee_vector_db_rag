import sys
import time
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from config.settings import settings

class GenerationService:
    """Handles LLM interaction and RAG logic with direct streaming and performance metrics."""

    def __init__(self, retriever):
        self.llm = OllamaLLM(
            base_url=settings.ollama_url,
            model=settings.ollama_llm_model,
            keep_alive="5m",
            streaming=True
        )
        self.retriever = retriever
        self.prompt_template = PromptTemplate.from_template("""
Use the following context to answer the user's question. 
If you don't know the answer based on the context, just say you don't know. 
Do not try to make up an answer.

Context:
{context}

Question: {question}

Answer:
""")

    def _format_context(self, docs: List[Any]) -> str:
        """Formats a list of documents into a single context string."""
        return "\n\n".join([f"--- Source: {d.metadata.get('filename', 'Unknown')} ---\n{d.page_content}" for d in docs])

    def run_with_metrics(self, query: str):
        """Runs the manual RAG pipeline and returns result with performance metrics."""
        
        # 1. Retrieval Phase
        print("[*] Retrieving relevant context...")
        docs = self.retriever.invoke(query)
        retrieval_time = getattr(self.retriever, "last_retrieval_time", 0.0)
        
        context_text = self._format_context(docs)
        
        # 2. Prompt Preparation
        final_prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )
        
        # 3. Generation Phase with Streaming
        print(f"\n[*] Generating LLM response...")
        print("-" * 30 + "\nANSWER:")
        
        tokens = []
        llm_start_time = None
        llm_end_time = None
        
        # We time the actual streaming start to end
        op_start_time = time.perf_counter()
        
        for chunk in self.llm.stream(final_prompt):
            if llm_start_time is None:
                llm_start_time = time.perf_counter()
            
            sys.stdout.write(chunk)
            sys.stdout.flush()
            tokens.append(chunk)
            
        llm_end_time = time.perf_counter()
        op_end_time = time.perf_counter()
        
        print("\n" + "-" * 30)
        
        # 4. Metrics Calculation
        total_duration = op_end_time - op_start_time
        llm_duration = (llm_end_time - llm_start_time) if llm_start_time else total_duration
        
        full_response = "".join(tokens)
        token_count = len(tokens)
        tps = token_count / llm_duration if llm_duration > 0 else 0
        
        metrics = {
            "total_time": total_duration + retrieval_time,
            "retrieval_time": retrieval_time,
            "llm_time": llm_duration,
            "tps": tps,
            "token_count": token_count
        }
        
        return {"result": full_response, "source_documents": docs}, metrics
