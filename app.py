"""
Flask web application for the RAG Orchestrator - Movie & TV Plot QA
Multi-agent system with LoRA fine-tuned generator for factual questions.

Run:
    python app.py
Then open http://localhost:8080
"""

import os
import json
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

os.environ["OPENAI_API_KEY"]   = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ── Globals ───────────────────────────────────────────────────────────────────
_initialized       = False
llm                = None
retriever          = None
prompt             = None
reranker_model     = None
raw_text           = {}
orchestrator_graph = None
lora_model         = None
lora_tokenizer     = None


def initialize_pipeline():
    global _initialized, llm, retriever, prompt, reranker_model
    global raw_text, orchestrator_graph, lora_model, lora_tokenizer

    if _initialized:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.retrievers import BaseRetriever
        from langchain_openai import ChatOpenAI
        from langchain_pinecone import PineconeVectorStore
        from sentence_transformers import CrossEncoder
        from langsmith import Client

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # ── LoRA model ────────────────────────────────────────────────────────
        BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        PT_PATH         = os.path.join(BASE_DIR, "smollm2-lora-narrativeqa.pt")

        print("Loading LoRA model on CPU...")
        lora_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        base_model_for_lora = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        lora_model = get_peft_model(base_model_for_lora, lora_config)
        state_dict = torch.load(PT_PATH, map_location="cpu")
        lora_model.load_state_dict(state_dict)
        lora_model.eval()
        print("✓ LoRA model loaded")

        # ── Embeddings & Vector Store ─────────────────────────────────────────
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vectorstore = PineconeVectorStore(
            index_name="stream-embeddings",
            embedding=embeddings,
            text_key="id",
            pinecone_api_key=PINECONE_API_KEY,
        )
        time.sleep(5)
        print("✓ Vector store initialised")

        # ── Raw chunks ────────────────────────────────────────────────────────
        chunks_path = os.path.join(BASE_DIR, "chunk_data", "merged_chunks.json")
        print(f"Loading chunks from: {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_text = {item["id"]: item["text"] for item in data}
        print(f"✓ Loaded {len(raw_text):,} chunks into memory")

        # ── LLM ───────────────────────────────────────────────────────────────
        llm = ChatOpenAI(model="gpt-4o-mini", seed=0)

        # ── Custom Retriever ──────────────────────────────────────────────────
        class CustomRetriever(BaseRetriever):
            vectorstore: PineconeVectorStore
            splits: Dict[str, str]

            def _get_relevant_documents(self, query: str) -> list:
                docs = self.vectorstore.similarity_search(query, k=10)
                results = []
                for doc in docs:
                    doc_id = str(doc.id) if doc.id is not None else None
                    if doc_id and doc_id in self.splits:
                        results.append(
                            Document(
                                page_content=self.splits[doc_id],
                                metadata={"id": doc.id, "title": doc.metadata.get("title", "")},
                            )
                        )
                return results

        retriever = CustomRetriever(vectorstore=vectorstore, splits=raw_text)

        client = Client()
        prompt = client.pull_prompt("rlm/rag-prompt")

        # ── Reranker ──────────────────────────────────────────────────────────
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # ── Graph ─────────────────────────────────────────────────────────────
        orchestrator_graph = _build_graph(
            llm, retriever, prompt, reranker_model,
            lora_model, lora_tokenizer, vectorstore, raw_text
        )

        _initialized = True
        print("✓ Pipeline initialized")

    except Exception:
        print("✗ Pipeline initialization failed:\n", traceback.format_exc())
        raise


def _build_graph(llm, retriever, prompt, reranker_model,
                 lora_model, lora_tokenizer, vectorstore, raw_text):
    import json as _json
    import torch as _torch
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser as SOP
    from langgraph.graph import END, StateGraph
    from typing import Optional as Opt

    # ── Types ──────────────────────────────────────────────────────────────
    class _QType(str, Enum):
        FACTUAL   = "factual"
        EMOTIONAL = "emotional"
        VAGUE     = "vague"

    # ── Helpers ────────────────────────────────────────────────────────────
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def rerank_chunks(input_dict, top_k=5, title_weight=0.2):
        query  = input_dict["hyde_query"]
        chunks = [d for d in input_dict["retrieved_chunks"] if isinstance(d, Document)]
        if not chunks:
            return []
        pairs  = [(query, doc.page_content) for doc in chunks]
        scores = reranker_model.predict(pairs)
        query_lower = query.lower()
        final_scores = [
            float(s) + (title_weight if doc.metadata.get("title","").lower() in query_lower else 0.0)
            for s, doc in zip(scores, chunks)
        ]
        ranked = sorted(zip(final_scores, chunks), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]

    def generate_with_lora(context: str, question: str, max_new_tokens: int = 64) -> str:
        full_prompt = (
            f"Summary (context):\n{context}\n\n"
            f"Question: {question}\n\n"
            "Please answer the question in one short sentence. "
            "Do not list options, only give the direct answer.\n\nAnswer:"
        )
        messages = [{"role": "user", "content": full_prompt}]
        inputs = lora_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(lora_model.device)

        with _torch.no_grad():
            output_ids = lora_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=lora_tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer    = lora_tokenizer.decode(generated, skip_special_tokens=True).strip()
        answer    = answer.split("\n")[0].strip()
        return answer if answer else "I don't know."

    def classify_and_rewrite(question: str):
        pt = f"""You are helping a retrieval system for movie and TV plot questions.

Given the user question below, do two things:

1. Classify it as exactly one of: factual | emotional | vague
   - factual: who/what/when/where questions about specific names, events, locations
   - emotional: why/how questions about feelings, motivations, relationships
   - vague: unclear, underspecified, or missing key details

2. Rewrite it as a short plot-focused statement suitable for vector search.
   Keep character names, time period, key events, causes and results.
   Do not summarize emotionally. Do not add additional information.

Respond with ONLY a JSON object, no explanation, no markdown:
{{"question_type": "factual|emotional|vague", "hyde_query": "rewritten statement here"}}

User question: "{question}"
"""
        raw = (llm | SOP()).invoke(pt)
        try:
            parsed     = _json.loads(raw)
            q_type_str = parsed.get("question_type", "factual").strip().lower()
            hyde_query = parsed.get("hyde_query", question).strip()
            if "emotional" in q_type_str:
                q_type = _QType.EMOTIONAL
            elif "vague" in q_type_str:
                q_type = _QType.VAGUE
            else:
                q_type = _QType.FACTUAL
        except Exception:
            q_type     = _QType.FACTUAL
            hyde_query = question
        return q_type, hyde_query

    def extract_or_infer_title(question, chunks):
        candidate_titles = list({
            doc.metadata.get("title", "") for doc in chunks
            if doc.metadata.get("title", "")
        })
        if not candidate_titles:
            return None
        candidates_str = "\n".join(f"- {t}" for t in candidate_titles)
        title_prompt = f"""You are given a user question about a movie or TV show and a list of
candidate titles from search results. Identify the single title the question is about.

Rules:
1. If the question directly mentions one of the candidates, return that title EXACTLY.
2. If not in the list but you know the title, return your best guess.
3. If general and not tied to one title, return NONE.

Respond with ONLY the title string or NONE. No explanation.

User question: "{question}"
Candidate titles:
{candidates_str}
"""
        raw_title = (llm | SOP()).invoke(title_prompt).strip()
        return None if raw_title.upper() == "NONE" or not raw_title else raw_title

    def filter_chunks_by_title(chunks, target_title, min_matching_chunks=2):
        target_lower    = target_title.lower()
        matching_chunks = [
            doc for doc in chunks
            if doc.metadata.get("title", "").lower() == target_lower
        ]
        if len(matching_chunks) >= min_matching_chunks:
            return matching_chunks, True
        return chunks, False

    def retrieve_with_title_filter(hyde_query, target_title):
        filtered_docs = vectorstore.similarity_search(
            hyde_query, k=10,
            filter={"title": {"$eq": target_title}},
        )
        return [
            Document(
                page_content=raw_text[str(doc.id)],
                metadata={"id": doc.id, "title": doc.metadata["title"]},
            )
            for doc in filtered_docs
            if str(doc.id) in raw_text
        ]

    def critic_check(answer, docs):
        snippets = [
            f"TITLE: {d.metadata.get('title','Unknown')}\n{d.page_content}"
            for d in docs[:5]
        ]
        ctx = "\n\n---\n\n".join(snippets)
        cp = f"""You are a strict fact-checker for answers about movie and TV plots.
You are given:
1) An answer proposed by another model
2) A set of plot snippets that must be treated as the only ground-truth evidence.

Decide whether the answer is fully supported by the snippets. If not, rewrite a corrected answer.

Respond ONLY with a JSON object, no explanation, no markdown:
{{"supported": true|false, "explanation": "...", "revised_answer": "..."}}

Answer to check:
{answer}

Evidence snippets:
{ctx}
"""
        raw = (llm | SOP()).invoke(cp)
        try:
            return _json.loads(raw)
        except Exception:
            return {
                "supported": False,
                "explanation": "Failed to parse critic JSON; treating as unsupported.",
                "revised_answer": answer,
            }

    def select_evidence_snippets(answer, chunks, max_snippets=3):
        return [
            {
                "title":   doc.metadata.get("title", ""),
                "id":      str(doc.metadata.get("id", "")),
                "snippet": doc.page_content[:200],
            }
            for doc in chunks[:max_snippets]
        ]

    # ── Nodes ──────────────────────────────────────────────────────────────
    def node_classify(state):
        q_type, hyde_query = classify_and_rewrite(state["question"])
        return {**state, "question_type": q_type.value, "hyde_query": hyde_query}

    def node_generate_factual(state):
        """LoRA Generator — factual questions"""
        hyde_query = state["hyde_query"]
        retrieved  = retriever.invoke(hyde_query)
        top_chunks = rerank_chunks({"hyde_query": hyde_query, "retrieved_chunks": retrieved})
        context    = format_docs(top_chunks)
        answer     = generate_with_lora(context, state["question"])
        return {
            **state,
            "strategy":       "lora_rewrite_rerank",
            "answer":         answer,
            "chunks":         top_chunks,
            "generator_used": "lora",
        }

    def node_generate_emotional(state):
        """GPT Generator — emotional questions"""
        hyde_query = state["hyde_query"]
        retrieved  = retriever.invoke(hyde_query)
        context    = format_docs(retrieved)
        answer     = (prompt | llm | SOP()).invoke(
            {"context": context, "question": state["question"]}
        )
        return {
            **state,
            "strategy":       "rewrite_only",
            "answer":         answer,
            "chunks":         retrieved,
            "generator_used": "gpt",
        }

    def node_generate_vague(state):
        """GPT Generator — vague questions with clarification"""
        clarify_prompt = (
            "You help rewrite vague questions about movie and TV plots. "
            "Given a possibly unclear question, rewrite it into a precise question "
            "that could be answered from a single episode or movie plot. "
            "Keep the same characters and setting if mentioned. Return only the rewritten question."
        )
        clarified  = (llm | SOP()).invoke(
            f"{clarify_prompt}\n\nOriginal question: {state['question']}"
        )
        _, hyde_query = classify_and_rewrite(clarified)
        retrieved     = retriever.invoke(hyde_query)
        top_chunks    = rerank_chunks({"hyde_query": hyde_query, "retrieved_chunks": retrieved})
        context       = format_docs(top_chunks)
        answer        = (prompt | llm | SOP()).invoke(
            {"context": context, "question": state["question"]}
        )
        return {
            **state,
            "strategy":           "clarify_then_rerank",
            "answer":             answer,
            "clarified_question": clarified,
            "hyde_query":         hyde_query,
            "chunks":             top_chunks,
            "generator_used":     "gpt",
        }

    def node_title_filter(state):
        chunks       = state.get("chunks", [])
        target_title = extract_or_infer_title(state["question"], chunks)
        if target_title is None:
            return {**state, "resolved_title": None, "title_filter_success": True}
        filtered_chunks, success = filter_chunks_by_title(chunks, target_title)
        return {
            **state,
            "chunks":               filtered_chunks,
            "resolved_title":       target_title,
            "title_filter_success": success,
        }

    def node_re_retrieve(state):
        hyde_query     = state.get("hyde_query", state["question"])
        resolved_title = state.get("resolved_title")

        if not resolved_title:
            chunks = retriever.invoke(hyde_query)
        else:
            chunks = retrieve_with_title_filter(hyde_query, resolved_title)

        if not chunks:
            return {
                **state,
                "title_filter_success": True,
                "retry_count":          state.get("retry_count", 0) + 1,
            }

        top_chunks = rerank_chunks({"hyde_query": hyde_query, "retrieved_chunks": chunks})
        context    = format_docs(top_chunks)

        if state.get("question_type") == _QType.FACTUAL.value:
            final_answer   = generate_with_lora(context, state["question"])
            generator_used = "lora"
        else:
            final_answer   = (prompt | llm | SOP()).invoke(
                {"context": context, "question": state["question"]}
            )
            generator_used = "gpt"

        return {
            **state,
            "chunks":               top_chunks,
            "answer":               final_answer,
            "title_filter_success": True,
            "retry_count":          state.get("retry_count", 0) + 1,
            "generator_used":       generator_used,
        }

    def node_critic(state):
        critic_result = critic_check(state["answer"], state.get("chunks", []))
        final_answer  = critic_result.get("revised_answer") or state["answer"]
        return {**state, "answer": final_answer, "critic_judgment": critic_result}

    def node_evidence(state):
        evidence = select_evidence_snippets(
            state["answer"], state.get("chunks", []), max_snippets=3
        )
        return {**state, "evidence": evidence}

    # ── Routing ────────────────────────────────────────────────────────────
    def route_after_classify(state):
        q_type = state.get("question_type", "factual")
        if q_type == _QType.EMOTIONAL.value:
            return "generate_emotional"
        elif q_type == _QType.VAGUE.value:
            return "generate_vague"
        else:
            return "generate_factual"

    def route_after_title_filter(state):
        if state.get("title_filter_success", False):
            return "critic"
        if state.get("retry_count", 0) >= 2:
            return "critic"
        return "re_retrieve"

    # ── Assemble ───────────────────────────────────────────────────────────
    g = StateGraph(dict)
    g.add_node("classify",           node_classify)
    g.add_node("generate_factual",   node_generate_factual)
    g.add_node("generate_emotional", node_generate_emotional)
    g.add_node("generate_vague",     node_generate_vague)
    g.add_node("title_filter",       node_title_filter)
    g.add_node("re_retrieve",        node_re_retrieve)
    g.add_node("critic",             node_critic)
    g.add_node("evidence",           node_evidence)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "generate_factual":   "generate_factual",
            "generate_emotional": "generate_emotional",
            "generate_vague":     "generate_vague",
        }
    )

    g.add_edge("generate_factual",   "title_filter")
    g.add_edge("generate_emotional", "title_filter")
    g.add_edge("generate_vague",     "title_filter")

    g.add_conditional_edges(
        "title_filter",
        route_after_title_filter,
        {"critic": "critic", "re_retrieve": "re_retrieve"}
    )

    g.add_edge("re_retrieve", "critic")
    g.add_edge("critic",      "evidence")
    g.add_edge("evidence",    END)

    return g.compile()


def run_orchestrator(query: str) -> dict:
    initialize_pipeline()
    print(f"\n→ Query: {query}")
    final = orchestrator_graph.invoke({"question": query, "retry_count": 0})
    return {
        "answer":          final.get("answer", ""),
        "question_type":   final.get("question_type", "factual"),
        "strategy":        final.get("strategy", ""),
        "generator_used":  final.get("generator_used", ""),
        "evidence":        final.get("evidence", []),
        "critic_judgment": final.get("critic_judgment", {}),
    }


# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

_HTML = open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html"),
    encoding="utf-8"
).read()


@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/api/query", methods=["POST"])
def api_query():
    body  = request.get_json(force=True)
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        result = run_orchestrator(query)
        return jsonify(result)
    except Exception:
        tb = traceback.format_exc()
        print("\n✗ ERROR:\n", tb)
        return jsonify({"error": tb}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")
