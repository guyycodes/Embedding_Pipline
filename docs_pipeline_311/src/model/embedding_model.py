import logging

import torch
import torch.nn.functional as F

from src.hugging_face_embedding import HuggingFaceModel
from src.util.get_agent_config import load_agent_config
from src.vector_store.qdrant_config import QdrantManager
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

class EmbeddingModel(HuggingFaceModel):
    """
    An embedding model that extends HuggingFaceModel.
    It tokenizes input texts, performs a forward pass through the model,
    applies average pooling to obtain embeddings, and additionally provides
    logic to upsert the embeddings into a Qdrant collection.
    """
    def __init__(self, model_name: str = None, device: str = None):
        super().__init__(model_name=model_name, device=device)
        # Load additional embedding-specific parameters from config.
        config = load_agent_config()
        agent_cfg = config.get("agent", {})
        embedding_cfg = agent_cfg.get("embedding_model", {})
        self.batch_size = embedding_cfg.get("batch_size", 32)
        self.max_length = embedding_cfg.get("max_tokens", 4096)
        
        # Setup Qdrant configuration.
        qdrant_cfg = config.get("qdrant", {})
        self.qdrant_collection = qdrant_cfg.get("collection", "document_vectors")
        self.qdrant_manager = QdrantManager(self.qdrant_collection)

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply average pooling over the token embeddings using the attention mask.
        """
        masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        return summed / counts.unsqueeze(1)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds a list of text chunks. Each chunk is assumed to be already
        prefixed with "passage: " if needed. This method:
          - batches them
          - tokenizes w/ self.tokenizer
          - passes them through self.model
          - uses average_pool for E5
          - returns a list of float vectors
        """
        all_embeddings = []
        
        for start_idx in range(0, len(documents), self.batch_size):
            batch_texts = documents[start_idx : start_idx + self.batch_size]

            # Debug: print out the chunk text (shortened)
            print(f"\n[DEBUG] Embedding batch from index {start_idx} => {start_idx + len(batch_texts)-1}")
            for i, txt in enumerate(batch_texts):
                preview = (txt[:60] + "...") if len(txt) > 60 else txt
                print(f"   chunk {start_idx + i}: '{preview}'")

            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Convert to float
            if "attention_mask" in encoding:
                print(f"[DEBUG] Encoding shape: input_ids={encoding['input_ids'].shape}, "
                      f"attention_mask={encoding['attention_mask'].shape}")
                encoding["attention_mask"] = encoding["attention_mask"].float()

            # Move to device
            for k, v in encoding.items(): 
                encoding[k] = v.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoding, return_dict=True)
                embeddings = self.average_pool(outputs.last_hidden_state, encoding["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # Convert to list of lists
            embeddings = embeddings.cpu().numpy().tolist()
            all_embeddings.extend(embeddings)

        return all_embeddings
            
    def embed_document(self, file_path: str) -> str:
        """
        Read a .txt file that contains multiple '*$%pass:' sections. 
        Each chunk of text after '*$%pass:' is considered one entry to embed.
        Embed and upsert them into Qdrant, then return a short result message.
        """
        try:
            # Make sure our collection is ready
            QdrantManager.ensure_qdrant_collection(self.qdrant_collection)
            
            # Step 1: Read the file contents
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            # Step 2: Split the file contents on the delimiter "*$%pass:"
            raw_chunks = full_text.split("*$%pass:")
            
            # Clean/trim and skip empty splits
            chunk_texts = []
            for chunk in raw_chunks:
                chunk = chunk.strip()
                if chunk:  # skip blank or empty splits
                    # Re-attach 'passage:' prefix for E5 usage
                    chunk_texts.append(f"passage: {chunk}")

            if not chunk_texts:
                logger.warning(f"No passages found in {file_path}. Nothing to embed.")
                return f"No passages found in {file_path}. Nothing to embed."
            
            # Step 3: Embed those chunks
            embeddings = self.embed_documents(chunk_texts)

            # Step 4: Prepare the points to upsert into Qdrant
            points = []
            for i, (vector, text) in enumerate(zip(embeddings, chunk_texts)):
                points.append(
                    PointStruct(
                        id=i,  # or a UUID
                        vector=vector,
                        payload={
                            "chunk_id": i,
                            "source_file": str(file_path),
                            "text": text
                        }
                    )
                )

            # Step 5: Upsert in batches (with logs)
            self.qdrant_manager.upsert_points(self.qdrant_collection, points, batch_size=200)

            # Step 6: Log + return success
            logger.info(
                f"[✨] Upserted {len(points)} chunks into '{self.qdrant_collection}'.\n"
                "Embedding ingestion complete."
            )
            return f"Successfully embedded document {file_path} with {len(embeddings)} embeddings."

        except Exception as e:
            logger.error(f"Error in embed_document for file {file_path}: {e}")
            return f"Error embedding document: {e}"
        
    def warm_up(self):
        logger.info("Skipping line-based embedding at warm-up; only loading model.")
        # You could do a trivial forward pass here if desired.
        pass