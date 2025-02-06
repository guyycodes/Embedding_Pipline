import logging
from typing import List

import torch
import torch.nn.functional as F

from qdrant_client.models import ScoredPoint  # for search results
from src.vector_store.qdrant_config import QdrantManager
from src.util.get_agent_config import load_agent_config
from src.hugging_face_query import HuggingFaceQuery  # Adjust this import path as needed

logger = logging.getLogger(__name__)

class QueryModel(HuggingFaceQuery):
    """
    Query model that extends HuggingFaceQuery.
    Provides:
      - A method to embed queries (with "query:" prefix).
      - A QA-style search method (qa_search) that embeds the query and searches Qdrant.
      - A deep semantic search method (deep_semantic_search) that returns a short snippet plus score.
    """

    def __init__(self, model_name: str = None, device: str = None):
        super().__init__(model_name=model_name, device=device)

        # Load query-specific settings from config.
        config = load_agent_config()
        agent_cfg = config.get("agent", {})
        query_cfg = agent_cfg.get("query_model", {})

        # read other settings for the QueryModel
        self.max_length = query_cfg.get("max_tokens", 4096)
        self.top_k = query_cfg.get("top_k", 5)

        # Optionally define batch_size for queries (if you want)
        self.batch_size = query_cfg.get("batch_size", 8)
        self.device = query_cfg.get("device", "cpu")
        # You can store 'timeout' or other fields if needed
        self.timeout = query_cfg.get("timeout", 10)

        logger.info(
            f"Initialized QueryModel with model_name={self.model_name}, device={self.device}, "
            f"max_length={self.max_length}, top_k={self.top_k}, batch_size={self.batch_size}"
        )

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply average pooling on the model's last hidden states using the attention mask.
        """
        masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        return summed / counts.unsqueeze(1)

    def embed_query(self, queries: List[str]) -> List[List[float]]:
        """
        Embed a list of queries.
        Each query is prefixed with "query:" (if required by your model)
        and processed in batches. Returns a list of normalized embedding vectors.
        """
        prefixed_queries = [f"query: {q}" for q in queries]
        all_embeddings = []

        for start_idx in range(0, len(prefixed_queries), self.batch_size):
            batch_queries = prefixed_queries[start_idx : start_idx + self.batch_size]
            logger.debug(f"Embedding queries from index {start_idx} to {start_idx + len(batch_queries)-1}")

            encoding = self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            if "attention_mask" in encoding:
                encoding["attention_mask"] = encoding["attention_mask"].float()

            for key, tensor_val in encoding.items():
                encoding[key] = tensor_val.to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoding, return_dict=True)
                embeddings = self.average_pool(outputs.last_hidden_state, encoding["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def qa_search(self, question: str, collection_name: str) -> List[str]:
        """
        1. Embed the user query.
        2. Search Qdrant for documents similar to the query in vector space.
        3. Optionally pass the question plus retrieved docs to a QA model (not shown).
        4. Return the retrieved document texts.
        """
        # ---------- RE-LOAD CONFIG AND OVERRIDE COLLECTION_NAME ------------------------
        config = load_agent_config()
        qdrant_cfg = config.get("qdrant", {})
        # if you want to force it to always use the YAML setting:
        collection_name = qdrant_cfg.get("collection", collection_name)
        # -------------------------------------------------------------------------------
        
        query_vector = self.embed_query([question])[0]
        if not query_vector:
            logger.info("Failed to create embedding for the query.")
            return ["I'm sorry, I couldn't understand that question."]

        # For top_k, we can read from config again or just use self.top_k
        top_k = self.top_k

        manager = QdrantManager(collection_name)
        search_results = manager.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=True  
            )  # returns List[ScoredPoint]

        if not search_results:
            logger.info("No matches found in Qdrant for Q/A search.")
            return ["No relevant context found."]

        retrieved_docs = []
        for i, point in enumerate(search_results):
            payload = point.payload
            if "text" in payload:
                logger.debug(f"Document {i + 1} text: {payload['text']}")
                retrieved_docs.append(payload["text"])

        return retrieved_docs

    def deep_semantic_search(self, query: str, collection_name: str) -> List[str]:
        """
        1. Embed the query.
        2. Search Qdrant for the top_k matches.
        3. Return a short snippet of each matched text along with its score.
        """
        # ---------- RE-LOAD CONFIG AND OVERRIDE COLLECTION_NAME -----------------------
        config = load_agent_config()
        qdrant_cfg = config.get("qdrant", {})
        collection_name = qdrant_cfg.get("collection", collection_name)
        # -------------------------------------------------------------------------------
        
        query_vector = self.embed_query([query])[0]
        if not query_vector:
            logger.info("Failed to create embedding for the query.")
            return ["I'm sorry, I couldn't understand that question."]

        # Use self.top_k here as well
        top_k = self.top_k

        manager = QdrantManager(collection_name)
        search_results = manager.search(query_vector, top_k=top_k)  # returns List[ScoredPoint]

        if not search_results:
            logger.info("No matches found in Qdrant for semantic search.")
            return ["No relevant context found."]

        output_list = []
        for hit in search_results:
            score = hit.score or 0.0
            text = hit.payload.get("text", "")
            snippet = text.replace("\n", " ")  # or keep newlines if you prefer
            output_list.append(f"[SCORE: {score:.4f}] {snippet}")

        return output_list

    def warm_up(self):
        """
        Perform a trivial query embedding to load/cache the model on the correct device.
        """
        logger.info("Warming up the query model with a trivial query embedding...")
        _ = self.embed_query(["Hello"])
        logger.info("Query model warm-up complete.")
