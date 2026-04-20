from typing import List, Dict, Any, Optional
from endee import Endee
from config.settings import settings

class DatabaseService:
    """Handles interaction with the Endee Vector Database."""

    def __init__(self):
        self.client = Endee()
        self.index_name = settings.endee_index_name
        self.dimension = settings.dense_dim
        self.space_type = settings.space_type

    def ensure_index(self, recreate: bool = False):
        """Ensures the index exists. Optionally recreates it."""
        if recreate:
            print(f"[*] Recreating index: {self.index_name}")
            try:
                self.client.delete_index(self.index_name)
            except Exception:
                pass
        else:
            # Check if it already exists
            try:
                self.client.get_index(self.index_name)
                print(f"[*] Index '{self.index_name}' already exists. Skipping creation.")
                return
            except Exception:
                # If get_index fails, we assume it needs to be created
                print(f"[*] Index '{self.index_name}' not found. Creating...")

        try:
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                space_type=self.space_type,
                M=settings.endee_m,
                ef_con=settings.endee_ef_con,
                precision=settings.endee_precision,
                sparse_model="endee_bm25",
            )
            print(f"[+] Index '{self.index_name}' created successfully.")
        except Exception as e:
            print(f"[!] Error during index creation: {e}")

    def get_index(self):
        """Returns the index object."""
        return self.client.get_index(self.index_name)

    def upsert_batch(self, index, points: List[Dict[str, Any]]):
        """Upserts a batch of points to the index."""
        if not points:
            return
        index.upsert(points)

    def query(self, index, vector: List[float], sparse_indices: List[int], sparse_values: List[float], top_k: int = 5, filt: Optional[List[dict]] = None):
        """Performs a hybrid query on the index."""
        kwargs = {
            "vector": vector,
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
            "top_k": top_k,
        }
        if filt:
            kwargs["filter"] = filt
        
        return index.query(**kwargs)
