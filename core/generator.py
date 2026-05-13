import sys
import time
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from core.semantic_cache import RedisSemanticCache
from config.settings import settings

class GenerationService:
    """Handles LLM interaction and RAG logic with direct streaming and performance metrics."""

    def __init__(self, retriever):
        self.llm = OllamaLLM(
            base_url=settings.ollama_url,
            model=settings.ollama_llm_model,
            temperature=settings.llm_temperature,
            keep_alive="5m",
            streaming=True
        )
        
        # We use ChatOllama for the query rewriting part as it handles message history better
        self.chat_llm = ChatOllama(
            base_url=settings.ollama_url,
            model=settings.ollama_llm_model,
            temperature=0, # Lower temperature for query rewriting
            keep_alive="5m"
        )
        
        self.base_retriever = retriever
        self.semantic_cache = RedisSemanticCache()
        
        # 1. Setup History-Aware Retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # This chain takes 'input' and 'chat_history' and returns a list of Documents
        self.history_aware_retriever = create_history_aware_retriever(
            self.chat_llm, self.base_retriever, contextualize_q_prompt
        )

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

    def run_with_metrics(self, query: str, chat_history: List[Any] = None, tenant_id: str = "default", doc_version: str = "1.0"):
        """Runs the manual RAG pipeline and returns result with performance metrics."""
        
        # 0. Query Normalization & Embedding Generation
        normalized_query = self.semantic_cache.normalize_query(query)
        q_dense = self.base_retriever.embedding_service.get_dense_embedding(normalized_query)

        # 1. Semantic Cache Lookup
        print("[*] Checking semantic cache...")
        cached_hit = self.semantic_cache.search(
            query_embedding=q_dense,
            tenant_id=tenant_id,
            doc_version=doc_version
        )

        if cached_hit and cached_hit.get("hit"):
            print("[+] Returning cached response.")
            return {
                "result": cached_hit["response"],
                "source_documents": [],
                "cache_hit": True,
                "similarity": cached_hit["similarity"]
            }, {
                "total_time": cached_hit["search_time"],
                "retrieval_time": 0.0,
                "llm_time": 0.0,
                "tps": 0.0,
                "token_count": len(cached_hit["response"].split()),
                "cache_hit": True
            }

        # 2. Retrieval Phase
        print("[*] Retrieving relevant context...")
        
        if chat_history:
            print(f"[*] Using history-aware retrieval for: {query}")
            # The history_aware_retriever returns Documents
            docs = self.history_aware_retriever.invoke({
                "input": query,
                "chat_history": chat_history
            })
        else:
            docs = self.base_retriever.invoke(query)
            
        retrieval_time = getattr(self.base_retriever, "last_retrieval_time", 0.0)
        
        context_text = self._format_context(docs)
        
        # 3. Prompt Preparation
        final_prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )
        
        # 4. Generation Phase with Streaming
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
        
        # 5. Metrics Calculation
        total_duration = op_end_time - op_start_time
        llm_duration = (llm_end_time - llm_start_time) if llm_start_time else total_duration
        
        full_response = "".join(tokens)
        token_count = len(tokens)
        tps = token_count / llm_duration if llm_duration > 0 else 0
        
        # 6. Store in Semantic Cache
        self.semantic_cache.store(
            query=normalized_query,
            embedding=q_dense,
            response=full_response,
            tenant_id=tenant_id,
            doc_version=doc_version
        )

        metrics = {
            "total_time": total_duration + retrieval_time,
            "retrieval_time": retrieval_time,
            "llm_time": llm_duration,
            "tps": tps,
            "token_count": token_count,
            "cache_hit": False
        }
        
        return {"result": full_response, "source_documents": docs, "cache_hit": False}, metrics
