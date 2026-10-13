from typing import Any, Dict, List, Optional

from core.config import CHROMA_DIR
from core.user_workspace import user_directory

# Chroma requires a vector per record; this placeholder is not used for search.
_STORAGE_VECTOR = [0.0]


def get_chroma_client() -> Any:
	import chromadb

	chroma_dir = user_directory("chroma", CHROMA_DIR)
	chroma_dir.mkdir(parents=True, exist_ok=True)
	return chromadb.PersistentClient(path=str(chroma_dir))


def get_collection(name: str = "idamp_memory") -> Any:
	return get_chroma_client().get_or_create_collection(
		name=name,
		embedding_function=None,
	)


def store_document(
	doc_id: str,
	text: str,
	metadata: Optional[Dict[str, Any]] = None,
) -> None:
	try:
		get_collection().upsert(
			ids=[str(doc_id)],
		documents=[text],
			metadatas=[metadata or {}],
			embeddings=[_STORAGE_VECTOR],
		)
	except Exception:
		pass


def query_memory(query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
	try:
		if n_results <= 0:
			return []

		results = get_collection().get(include=["documents", "metadatas"])
		ids = results.get("ids") or []
		documents = results.get("documents") or []
		metadatas = results.get("metadatas") or []
		needle = query_text.strip().casefold()
		matches = []

		for index, doc_id in enumerate(ids):
			document = documents[index] if index < len(documents) else None
			metadata = metadatas[index] if index < len(metadatas) else None
			if needle and (
				not isinstance(document, str) or needle not in document.casefold()
			):
				continue
			matches.append(
				{
					"id": doc_id,
					"document": document,
					"metadata": metadata or {},
				}
			)
			if len(matches) >= n_results:
				break

		return matches
	except Exception:
		return []