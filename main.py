from rag_retrieval import DocumentLoader

loader = DocumentLoader("crawler/result_data/rag_output")
documents = loader.load_all_documents()
print(f"Loaded {len(documents)} documents.")

# asyncio.run(crawler.execute_crawler())
