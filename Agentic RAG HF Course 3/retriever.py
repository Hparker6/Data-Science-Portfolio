from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool

guest_dataset = load_dataset("agents-course/unit3-invitees", split="train")

docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info",
    func=extract_text,
    description="Use this tool to retrieve background info about invited gala guests using their name or relation."
)
