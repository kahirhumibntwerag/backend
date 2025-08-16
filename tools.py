# arxiv_tools.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Optional, Dict, Literal

import arxiv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from qdrant_client import models
from qdrant import vector_store



# --------------------
# Data structures
# --------------------
@dataclass
class ArxivPaper:
    id: str                     # short id, e.g. "2501.12345"
    title: str
    summary: str
    authors: List[str]
    categories: List[str]
    published: Optional[str]    # ISO string
    updated: Optional[str]      # ISO string
    pdf_url: Optional[str]
    entry_id: str               # full Atom entry URL
    comment: Optional[str]
    doi: Optional[str]
    primary_category: Optional[str]
    links: Dict[str, str]       # label -> href


class ArxivClientWrapper:
    """
    Thin wrapper around the `arxiv` library client that:
    - Respects rate limits with delay_seconds
    - Retries transient errors (num_retries)
    - Converts results to a consistent ArxivPaper schema
    """
    def __init__(
        self,
        delay_seconds: float = 3.0,
        max_retries: int = 3,
        log_level: int = logging.INFO,
    ) -> None:
        self.client = arxiv.Client(num_retries=max_retries, delay_seconds=delay_seconds)
        logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    def _convert(self, r: arxiv.Result) -> ArxivPaper:
        return ArxivPaper(
            id=r.get_short_id(),
            title=(r.title or "").strip(),
            summary=(r.summary or "").strip(),
            authors=[a.name for a in r.authors] if r.authors else [],
            categories=list(getattr(r, "categories", [])),
            published=r.published.isoformat() if getattr(r, "published", None) else None,
            updated=r.updated.isoformat() if getattr(r, "updated", None) else None,
            pdf_url=getattr(r, "pdf_url", None),
            entry_id=r.entry_id,
            comment=getattr(r, "comment", None),
            doi=getattr(r, "doi", None),
            primary_category=getattr(r, "primary_category", None),
            links={(l.title or "link"): l.href for l in getattr(r, "links", [])},
        )

    def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "submittedDate",
        sort_order: Literal["ascending", "descending"] = "descending",
    ) -> List[ArxivPaper]:
        """
        Query syntax (most useful fields):
          - ti:    title text, e.g. ti:"diffusion model"
          - au:    author,      e.g. au:"Goodfellow"
          - abs:   abstract,    e.g. abs:"solar wind"
          - cat:   category,    e.g. cat:cs.LG OR cat:astro-ph.SR
          - Use AND/OR and parentheses to combine terms.
        """
        sb_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        so_map = {"ascending": arxiv.SortOrder.Ascending, "descending": arxiv.SortOrder.Descending}

        logging.info("Querying arXiv: %s | max_results=%d", query, max_results)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sb_map.get(sort_by, arxiv.SortCriterion.SubmittedDate),
            sort_order=so_map.get(sort_order, arxiv.SortOrder.Descending),
        )
        results = list(self.client.results(search))
        papers = [self._convert(r) for r in results]
        logging.info("Received %d results", len(papers))
        return papers

    def get_by_id(self, id_or_url: str) -> Optional[ArxivPaper]:
        """
        Accepts: "2501.12345", "arXiv:2501.12345", or a full arxiv.org URL.
        """
        short_id = id_or_url.split("/")[-1]
        if short_id.startswith("arXiv:"):
            short_id = short_id.split("arXiv:")[1]
        search = arxiv.Search(id_list=[short_id])
        results = list(self.client.results(search))
        return self._convert(results[0]) if results else None

    def download_pdf(self, id_or_url: str, dest_dir: str = "./downloads", filename: Optional[str] = None) -> str:
        """
        Download the paper's PDF by arXiv ID/URL. Returns the file path.
        """
        os.makedirs(dest_dir, exist_ok=True)

        short_id = id_or_url.split("/")[-1]
        if short_id.startswith("arXiv:"):
            short_id = short_id.split("arXiv:")[1]

        results = list(self.client.results(arxiv.Search(id_list=[short_id])))
        if not results:
            raise ValueError(f"No paper found for {id_or_url}")

        r = results[0]
        path = r.download_pdf(dirpath=dest_dir, filename=filename)
        logging.info("Saved PDF to %s", path)
        return path


# --------------------
# Singleton client (shared by tools)
# --------------------
@lru_cache(maxsize=1)
def _get_client() -> ArxivClientWrapper:
    delay = float(os.getenv("ARXIV_DELAY_SECONDS", "3.0"))
    retries = int(os.getenv("ARXIV_MAX_RETRIES", "3"))
    level = getattr(logging, os.getenv("ARXIV_LOG_LEVEL", "INFO").upper(), logging.INFO)
    return ArxivClientWrapper(delay_seconds=delay, max_retries=retries, log_level=level)


# --------------------
# Pydantic arg schemas for tools
# --------------------
class SearchArgs(BaseModel):
    query: str = Field(
        ...,
        description=(
            "arXiv query string. Examples:\n"
            '  ti:"diffusion model"\n'
            '  abs:"solar wind" AND cat:astro-ph.SR\n'
            '  (ti:diffusion OR abs:diffusion) AND cat:cs.LG'
        ),
    )
    max_results: int = Field(20, ge=1, le=200, description="Max number of papers to return (1–200).")
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(
        "submittedDate", description="Sort field: relevance | lastUpdatedDate | submittedDate"
    )
    sort_order: Literal["ascending", "descending"] = Field(
        "descending", description="Sort order: ascending | descending"
    )


class GetByIdArgs(BaseModel):
    id_or_url: str = Field(
        ...,
        description='arXiv id or URL (e.g., "2501.12345", "arXiv:2501.12345", or "https://arxiv.org/abs/2501.12345").',
    )


class DownloadArgs(BaseModel):
    id_or_url: str = Field(..., description="arXiv id or URL to download the PDF for.")
    dest_dir: str = Field("./downloads", description="Directory to save the PDF.")
    filename: Optional[str] = Field(None, description="Optional custom filename (e.g., 'paper.pdf').")


# --------------------
# LangChain tools
# --------------------
@tool("arxiv_search", args_schema=SearchArgs)
def arxiv_search(
    query: str,
    max_results: int = 20,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> List[Dict]:
    """Search arXiv and return a list of papers as JSON dictionaries."""
    client = _get_client()
    papers = client.search(query=query, max_results=max_results, sort_by=sort_by, sort_order=sort_order)
    return [asdict(p) for p in papers]


@tool("arxiv_get_by_id", args_schema=GetByIdArgs)
def arxiv_get_by_id(id_or_url: str) -> Optional[Dict]:
    """Fetch a single arXiv paper by id or URL and return it as a JSON dictionary."""
    client = _get_client()
    paper = client.get_by_id(id_or_url=id_or_url)
    return asdict(paper) if paper else None


@tool("arxiv_download_pdf", args_schema=DownloadArgs)
def arxiv_download_pdf(id_or_url: str, dest_dir: str = "./downloads", filename: Optional[str] = None) -> str:
    """Download the PDF of an arXiv paper and return the local file path."""
    client = _get_client()
    path = client.download_pdf(id_or_url=id_or_url, dest_dir=dest_dir, filename=filename)
    return path



@tool
def search_documents(query: str, config: RunnableConfig) -> str:
    """
    Search for relevant documents in the user's personal knowledge base.
    
    Args:
        query: The search query to find relevant documents
    
    Returns:
        A formatted string containing relevant documents and their scores
    """
    # Access configurable parameters correctly
    configurable = config.get("configurable", {})
    user = configurable.get("user")
    store_name = configurable.get("store_name")
    
    # Add validation
    if not user:
        return "Error: User information not available"
    
    if not store_name:
        return "Error: No store name specified. Please provide a store name."
    
    try:
        # Create filter for user's store
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user", 
                    match=models.MatchValue(value=user)
                ),
                models.FieldCondition(
                    key="metadata.store_name", 
                    match=models.MatchValue(value=store_name)
                ),
            ]
        )

        # Search for relevant documents
        try:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=5,
                filter=filter,
            )
            print(results)
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            return f"Error searching document store: {str(e)}"

        if results:
            print(f"Found {len(results)} documents for query: {query}")
            # Format the results
            formatted_docs = "\n\n".join([
                f"**Document {i+1} (Score: {score:.2f}):**\n{doc.page_content}"
                for i, (doc, score) in enumerate(results)
            ])
            
            return f"Found relevant documents from store '{store_name}':\n\n{formatted_docs}"
        else:
            return f"No relevant documents found in store '{store_name}' for the query: '{query}'"
        
    except Exception as e:
        print(f"Error in search_documents: {str(e)}")
        return f"Error searching document store: {str(e)}"