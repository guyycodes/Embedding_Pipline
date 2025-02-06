# main.py

import logging
import time
from threading import Thread, Lock, current_thread
from queue import Empty

from src.util.queue import FileQueue
from src.util.document_cleaning_pipline import DocumentCleaner
from src.util.clean_doc_queue import CleanDocQueue
from src.util. get_agent_config import load_agent_config
from initializers import warm_up_embedder, warm_up_query_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

### Configuration for the document cleaning pipeline ###
INITIAL_CONSUMER_COUNT = 2
MAX_CONSUMER_COUNT = 10
QUEUE_THRESHOLD = 100     # When the queue has >100 items, spawn extra consumers.
CONSUMER_TIMEOUT = 10     # Timeout in seconds for idle consumer threads.

consumer_threads = []
consumer_lock = Lock()
agnt_cofig = load_agent_config()
a = agnt_cofig.get("agent", {})
m = a.get("mode_name", "guymorganb/e5-large-v2-4096-lsg-patched")

def consumer_worker(file_queue: FileQueue, cleaner: DocumentCleaner):
    logger.info(f"Document cleaning consumer {current_thread().name} started.")
    while True:
        try:
            file_path = file_queue.queue.get(timeout=CONSUMER_TIMEOUT)
            logger.info(f"{current_thread().name} processing file: {file_path}")
            result = cleaner.process_document_from_queue(file_path)
            logger.info(f"{current_thread().name} result: {result}")
            file_queue.queue.task_done()
        except Empty:
            with consumer_lock:
                if len(consumer_threads) > INITIAL_CONSUMER_COUNT:
                    logger.info(f"{current_thread().name} exiting (idle).")
                    consumer_threads[:] = [t for t in consumer_threads if t.name != current_thread().name]
                    break
            continue
        except Exception as e:
            logger.error(f"{current_thread().name} encountered error: {e}")
            file_queue.queue.task_done()

def monitor_consumers(file_queue: FileQueue, cleaner: DocumentCleaner):
    while True:
        if file_queue.queue.qsize() > QUEUE_THRESHOLD:
            with consumer_lock:
                if len(consumer_threads) < MAX_CONSUMER_COUNT:
                    t = Thread(target=consumer_worker, args=(file_queue, cleaner), daemon=True)
                    consumer_threads.append(t)
                    t.start()
                    logger.info(f"Spawned cleaning consumer. Total: {len(consumer_threads)}")
        time.sleep(5)

### Configuration for the embedding pipeline ###
INITIAL_EMBEDDING_CONSUMER_COUNT = 2
MAX_EMBEDDING_CONSUMER_COUNT = 10
EMBEDDING_QUEUE_THRESHOLD = 100
EMBEDDING_CONSUMER_TIMEOUT = 10

embedding_consumer_threads = []
embedding_consumer_lock = Lock()

def embedding_consumer_worker(clean_doc_queue: CleanDocQueue, embedder):
    """
    Uses the warmed-up embedder to process the document.
    """
    logger.info(f"Embedding consumer {current_thread().name} started.")
    while True:
        try:
            file_path = clean_doc_queue.queue.get(timeout=EMBEDDING_CONSUMER_TIMEOUT)
            logger.info(f"{current_thread().name} processing clean doc: {file_path}")
            
            # Use the embedder to embed the document.
            # result = embedder.embed_document(file_path)
            # logger.info(f"{current_thread().name} embedded result: {result}")
            
            clean_doc_queue.queue.task_done()
        except Empty:
            with embedding_consumer_lock:
                if len(embedding_consumer_threads) > INITIAL_EMBEDDING_CONSUMER_COUNT:
                    logger.info(f"{current_thread().name} exiting (idle).")
                    embedding_consumer_threads[:] = [t for t in embedding_consumer_threads if t.name != current_thread().name]
                    break
            continue
        except Exception as e:
            logger.error(f"{current_thread().name} encountered error: {e}")
            clean_doc_queue.queue.task_done()

def monitor_embedding_consumers(clean_doc_queue: CleanDocQueue, embedder):
    while True:
        if clean_doc_queue.queue.qsize() > EMBEDDING_QUEUE_THRESHOLD:
            with embedding_consumer_lock:
                if len(embedding_consumer_threads) < MAX_EMBEDDING_CONSUMER_COUNT:
                    t = Thread(target=embedding_consumer_worker, args=(clean_doc_queue, embedder), daemon=True)
                    embedding_consumer_threads.append(t)
                    t.start()
                    logger.info(f"Spawned embedding consumer. Total: {len(embedding_consumer_threads)}")
        time.sleep(5)
        

def main():
    ### Start the Document Cleaning Pipeline ###
    file_queue = FileQueue()
    file_queue_monitor = Thread(target=file_queue.run, daemon=True)
    file_queue_monitor.start()
    logger.info("Started FileQueue monitoring for dirty documents.")
    
    cleaner = DocumentCleaner()
    for _ in range(INITIAL_CONSUMER_COUNT):
        t = Thread(target=consumer_worker, args=(file_queue, cleaner), daemon=True)
        with consumer_lock:
            consumer_threads.append(t)
        t.start()
    logger.info(f"Started {INITIAL_CONSUMER_COUNT} cleaning consumer threads.")
    
    scaling_thread = Thread(target=monitor_consumers, args=(file_queue, cleaner), daemon=True)
    scaling_thread.start()
    logger.info("Started document cleaning scaling monitor.")

    ### Start the Embedding Pipeline ###
    clean_doc_queue = CleanDocQueue()
    clean_doc_monitor = Thread(target=clean_doc_queue.run, daemon=True)
    clean_doc_monitor.start()
    logger.info("Started CleanDocQueue monitoring for clean documents.")
    
    # 1) Warm up the embedder
    embedder = warm_up_embedder(model_name=m)

    # 2) Warm up the query model
    query_model = warm_up_query_model(model_name=m) 
    
    for _ in range(INITIAL_EMBEDDING_CONSUMER_COUNT):
        t = Thread(target=embedding_consumer_worker, args=(clean_doc_queue, embedder), daemon=True)
        with embedding_consumer_lock:
            embedding_consumer_threads.append(t)
        t.start()
    logger.info(f"Started {INITIAL_EMBEDDING_CONSUMER_COUNT} embedding consumer threads.")
    
    embedding_scaling_thread = Thread(target=monitor_embedding_consumers, args=(clean_doc_queue, embedder), daemon=True)
    embedding_scaling_thread.start()
    logger.info("Started embedding consumer scaling monitor.")

    # Here you can also start your API server or other parts of your application.
    # the api server will take in uploads and settings chages for the pipline
    # For example, if using FastAPI, you might start it here or via another process.
    # This main loop simply keeps the program alive.
    while True:
        pass

if __name__ == "__main__":
    main()
