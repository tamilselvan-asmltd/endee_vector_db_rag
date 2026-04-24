from endee import Endee
import os
import sys

# Add current dir to sys.path
sys.path.append(os.getcwd())

try:
    client = Endee()
    index_name = "cnc_hybrid_vdb"
    try:
        index = client.get_index(index_name)
        print(f"Index object: {type(index)}")
        print(f"Methods: {dir(index)}")
    except Exception as e:
        print(f"Could not get index: {e}")
except Exception as e:
    print(f"Could not initialize Endee: {e}")
