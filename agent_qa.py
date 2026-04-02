"""
ReAct (Reasoning + Action) Agent for YouTube QA Bot
Uses Qwen.ai as the LLM provider via OpenAI-compatible API.

The agent reasons step-by-step and calls tools to search video content
before producing a final answer with timestamps and sources.
"""

import os
import json
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import lancedb
from sentence_transformers import SentenceTransformer

load_dotenv()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class YouTubeReActAgent:
    """
    ReAct agent that reasons over a LanceDB video transcript database.

    Tools available to the agent:
      - search_video_content   : semantic search across all videos
      - search_within_video    : semantic search inside one specific video
      - list_videos            : list all videos in the database
      - search_by_tags         : semantic search filtered to specific tags
    """

    def __init__(self):
        lancedb_path = os.getenv("LANCEDB_PATH", "lancedb")
        self.db = lancedb.connect(lancedb_path)
        self._embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._embedding_model: Optional[SentenceTransformer] = None
        self._llm: Optional[ChatOpenAI] = None

    # ------------------------------------------------------------------
    # Lazy-loaded models
    # ------------------------------------------------------------------

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            print(f"Loading embedding model: {self._embedding_model_name}")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            api_key = os.getenv("QWEN_API_KEY", "")
            base_url = os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
            model = os.getenv("QWEN_MODEL", "qwen-plus")
            temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
            max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))

            if not api_key:
                raise ValueError(
                    "QWEN_API_KEY is not set. "
                    "Add it to your .env file. Get a free key at https://dashscope.aliyuncs.com/"
                )

            self._llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Internal search helper
    # ------------------------------------------------------------------

    def _search_chunks(
        self,
        query: str,
        limit: int = 5,
        video_ids: List[str] = None,
        tags: List[str] = None,
    ) -> List[dict]:
        """Semantic search over video_chunks table with optional filters."""
        try:
            if "video_chunks" not in self.db.table_names():
                return []

            query_embedding = self.embedding_model.encode(query)
            table = self.db.open_table("video_chunks")
            df = table.to_pandas()

            if df.empty:
                return []

            df["emb_arr"] = df["embedding"].apply(
                lambda x: np.array(x) if isinstance(x, list) else x
            )

            scored = []
            for _, row in df.iterrows():
                sim = cosine_similarity(query_embedding, row["emb_arr"])
                scored.append((row, float(sim)))

            scored.sort(key=lambda x: x[1], reverse=True)

            results = []
            for row, sim in scored:
                # Filter by video_id whitelist
                if video_ids and row["video_id"] not in video_ids:
                    continue

                # Filter by tags
                if tags:
                    row_tags = row.get("tags", []) or []
                    if hasattr(row_tags, "tolist"):
                        row_tags = row_tags.tolist()
                    if not any(t in row_tags for t in tags):
                        continue

                results.append(
                    {
                        "video_id": row["video_id"],
                        "title": row["title"],
                        "text": row["text"],
                        "timestamp": row.get("timestamp", "N/A"),
                        "similarity": round(sim, 3),
                        "tags": (
                            row["tags"].tolist()
                            if hasattr(row.get("tags"), "tolist")
                            else (row.get("tags") or [])
                        ),
                    }
                )
                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            return [{"error": str(e)}]

    def _format_results(self, results: List[dict]) -> str:
        """Format search results into readable text for the agent."""
        if not results:
            return "No results found."
        if "error" in results[0]:
            return f"Search error: {results[0]['error']}"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[Result {i}] \"{r['title']}\" @ {r['timestamp']} "
                f"(video_id: {r['video_id']}, similarity: {r['similarity']})\n"
                f"{r['text']}"
            )
        return "\n\n---\n\n".join(lines)

    # ------------------------------------------------------------------
    # Tool functions
    # ------------------------------------------------------------------

    def tool_search_video_content(self, query: str) -> str:
        """Semantic search across all videos in the database."""
        results = self._search_chunks(query, limit=5)
        return self._format_results(results)

    def tool_search_within_video(self, input_str: str) -> str:
        """
        Search within a specific video.
        Input format: 'video_id | query'
        Example: 'dQw4w9WgXcQ | what is the main theme'
        """
        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "Error: Input must be 'video_id | query'. Use list_videos first to get video IDs."
        video_id = parts[0].strip()
        query = parts[1].strip()
        results = self._search_chunks(query, limit=5, video_ids=[video_id])
        return self._format_results(results)

    def tool_list_videos(self, _: str = "") -> str:
        """List all available videos with their IDs, chunk counts, and tags."""
        try:
            if "video_chunks" not in self.db.table_names():
                return "The video database is empty. No videos have been added yet."

            table = self.db.open_table("video_chunks")
            df = table.to_pandas()

            if df.empty:
                return "The video database is empty."

            # Aggregate per video
            summary = (
                df.groupby("video_id")
                .agg(title=("title", "first"), chunk_count=("chunk_index", "count"))
                .reset_index()
            )

            # Collect unique tags per video
            tag_map = {}
            for _, row in df.iterrows():
                vid = row["video_id"]
                t = row.get("tags", []) or []
                if hasattr(t, "tolist"):
                    t = t.tolist()
                tag_map.setdefault(vid, set()).update(t)

            lines = [f"Videos in database ({len(summary)} total):"]
            for _, row in summary.iterrows():
                tags = sorted(tag_map.get(row["video_id"], set()))
                tag_str = f"  [tags: {', '.join(tags)}]" if tags else ""
                lines.append(
                    f"  - \"{row['title']}\"\n"
                    f"    ID: {row['video_id']} | {row['chunk_count']} chunks{tag_str}"
                )
            return "\n".join(lines)

        except Exception as e:
            return f"Error listing videos: {str(e)}"

    def tool_search_by_tags(self, input_str: str) -> str:
        """
        Search video content filtered to specific tags.
        Input format: 'tag1,tag2 | query'
        Example: 'python,tutorial | how to use decorators'
        """
        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "Error: Input must be 'tag1,tag2 | query'."
        tags = [t.strip() for t in parts[0].split(",") if t.strip()]
        query = parts[1].strip()
        results = self._search_chunks(query, limit=5, tags=tags)
        return self._format_results(results)

    # ------------------------------------------------------------------
    # Agent setup
    # ------------------------------------------------------------------

    def _build_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_video_content",
                func=self.tool_search_video_content,
                description=(
                    "Semantically search ALL videos in the database for content "
                    "relevant to a query. Returns the most similar transcript segments "
                    "with timestamps and video titles. "
                    "Use this as your primary search tool. "
                    "Input: a plain-text search query."
                ),
            ),
            Tool(
                name="search_within_video",
                func=self.tool_search_within_video,
                description=(
                    "Search for content WITHIN a specific video by its video_id. "
                    "Use this when you already know which video to look in and want "
                    "more targeted results. "
                    "Input format: 'video_id | your query'  "
                    "(e.g. 'dQw4w9WgXcQ | what is the main point'). "
                    "Use list_videos first if you need to find a video_id."
                ),
            ),
            Tool(
                name="list_videos",
                func=self.tool_list_videos,
                description=(
                    "List all videos available in the database, showing their titles, "
                    "video IDs, chunk counts, and tags. "
                    "Use this when you need to discover what content is available, "
                    "find a specific video_id, or understand the scope of the library. "
                    "Input: any string (ignored)."
                ),
            ),
            Tool(
                name="search_by_tags",
                func=self.tool_search_by_tags,
                description=(
                    "Search video content filtered to videos that have specific tags. "
                    "Use this when the question is about a particular topic/category "
                    "that you know is tagged (e.g. 'python', 'ml', 'tutorial'). "
                    "Input format: 'tag1,tag2 | your query'."
                ),
            ),
        ]

    def _build_prompt(self) -> PromptTemplate:
        template = """You are an expert assistant that answers questions about YouTube video content.
You have access to a searchable database of video transcripts.
Think carefully step by step. Always search before answering — never guess.

You have access to the following tools:

{tools}

Use EXACTLY this format for every step:

Question: the input question you must answer
Thought: what do I need to do next?
Action: the tool to use — must be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought / Action / Action Input / Observation as needed)
Thought: I now have enough information to give a complete answer.
Final Answer: a comprehensive answer to the original question, citing specific video titles and timestamps from the Observations above.

Rules:
- Always use at least one tool before giving a Final Answer.
- If the database is empty, say so clearly.
- If you cannot find relevant content, say so — do not make up information.
- Include specific timestamps (e.g. "05:23 - 06:45") and video titles in your Final Answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        return PromptTemplate.from_template(template)

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(self, question: str, verbose: bool = True) -> str:
        """
        Run the ReAct agent on a question about video content.

        Args:
            question: Natural-language question about the video library.
            verbose:  If True, prints the Thought/Action/Observation trace.

        Returns:
            The agent's final answer string.
        """
        tools = self._build_tools()
        prompt = self._build_prompt()

        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=10,
            handle_parsing_errors=(
                "I encountered a formatting error. "
                "I will try again with the correct Thought/Action/Action Input format."
            ),
        )

        result = executor.invoke({"input": question})
        return result.get("output", "The agent did not produce an answer.")


# ------------------------------------------------------------------
# Standalone CLI entry point
# ------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ReAct agent for YouTube video QA (powered by Qwen.ai)"
    )
    parser.add_argument("question", help="Question to ask about the video library")
    parser.add_argument(
        "--quiet", action="store_true", help="Hide reasoning trace, show only final answer"
    )
    args = parser.parse_args()

    print(f"\nQuestion: {args.question}\n")
    if not args.quiet:
        print("=== Agent Reasoning Trace ===")

    agent = YouTubeReActAgent()
    answer = agent.run(args.question, verbose=not args.quiet)

    print("\n=== Final Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
