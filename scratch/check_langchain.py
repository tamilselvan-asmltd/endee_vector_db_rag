import langchain
print(f"LangChain version: {langchain.__version__}")
try:
    from langchain.chains import create_history_aware_retriever
    print("Successfully imported create_history_aware_retriever")
except ImportError as e:
    print(f"ImportError: {e}")

import sys
print(f"Python path: {sys.path}")
