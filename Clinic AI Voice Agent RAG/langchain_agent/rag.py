import sys

from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings


def retrieve(
    client: QdrantClient,
    embedder: SentenceTransformer,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Embed the query and return the top-k most similar chunks from Qdrant."""
    query_vector = embedder.encode(query).tolist()

    hits = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]


def build_context(retrieved_chunks: list[dict]) -> str:
    """Build a context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Source {i}]\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def rag(query: str, top_k: int = 5, client=None, embedder=None):
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks from Qdrant
      2. Format as context
      3. Send context + query to Groq LLM

    Returns:
        tuple: (answer_text, context_used)
    """
    if embedder is None:
        embedder = SentenceTransformer(settings.embedding_model)
    if client is None:
        client = QdrantClient(path=settings.qdrant_path)

    # Step 1 — Retrieve
    chunks = retrieve(client, embedder, query, top_k=top_k)
    if not chunks:
        return (
            "I couldn't find specific information about that in our documentation. "
            "Could you rephrase your question or ask about clinic information, appointments, or visit procedures?",
            "",
        )

    # Step 2 — Build context
    context = build_context(chunks)

    # Step 3 — Generate answer
    system_prompt = (
    "You are the voice assistant for Sahyadri Family Clinic, a fictional demo clinic.\n"
    "Answer using ONLY the retrieved clinic context. If information is missing, say so and offer human handoff.\n"
    "Never invent clinic facts, doctor credentials, appointment availability, or completed actions.\n"
    "You are not a clinician: do not diagnose, prescribe, change medication, or replace professional care.\n"
    "Medical guidance is draft-only unless the retrieved content explicitly says it is clinician-approved.\n"
    "For possible emergencies, do not delay care with questions; use the configured emergency escalation wording.\n"
    "Keep answers short, calm, respectful, and suitable for a voice call (usually 1-2 sentences)."
    )   

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0.2,
    )
    messages = [
        ("system", system_prompt),
        ("user", user_message),
    ]

    response = llm.invoke(messages)
    return response.content, context


def main():
    """CLI interface for testing RAG independently."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the clinic hours?"

    print(f"Question: {query}\n")
    print("Generating answer...\n")

    answer, context = rag(query)

    print(f"ANSWER:\n{answer}")
    print(f"\n{'='*60}")
    print(f"\nSOURCES:\n{context}")


if __name__ == "__main__":
    main()
