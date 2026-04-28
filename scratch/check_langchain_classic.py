try:
    from langchain_classic.chains import create_history_aware_retriever
    print("Successfully imported create_history_aware_retriever from langchain_classic")
except ImportError as e:
    print(f"ImportError (langchain_classic): {e}")

try:
    from langchain_classic.chains import create_retrieval_chain
    print("Successfully imported create_retrieval_chain from langchain_classic")
except ImportError as e:
    print(f"ImportError (langchain_classic): {e}")

try:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    print("Successfully imported create_stuff_documents_chain from langchain_classic")
except ImportError as e:
    print(f"ImportError (langchain_classic): {e}")
