"""
Singapore Compliance Assistant — Chat App
==========================================
Run with: streamlit run compliance_app.py
"""

import streamlit as st
import anthropic
import voyageai
from supabase import create_client
import re

# ── Config — keys loaded from Streamlit Secrets ───────────────────────────────
ANTHROPIC_KEY = st.secrets["ANTHROPIC_KEY"]
VOYAGE_KEY    = st.secrets["VOYAGE_KEY"]
SUPABASE_URL  = st.secrets["SUPABASE_URL"]
SUPABASE_KEY  = st.secrets["SUPABASE_KEY"]

SYSTEM_PROMPT = """You are a Singapore building compliance assistant for an architecture firm.
You have access to official Singapore authority codes: BCA, URA, SCDF, PUB, NEA, LTA, NParks.

Rules:
1. ONLY answer using the provided code excerpts. Never rely on memory for clause numbers.
2. Always cite exact clause reference e.g. "BCA Clause 4.2.1".
3. If authorities conflict, flag it. Most stringent requirement always applies.
4. If excerpts don't cover the question, say so clearly.
5. Format: Requirement → Clause reference → Conflicts or notes.
6. Be concise and practical for busy architects.
7. When analysing plans or images, identify dimensions and room labels, check against codes."""

# ── Clients ───────────────────────────────────────────────────────────────────

@st.cache_resource
def init_clients():
    c = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    v = voyageai.Client(api_key=VOYAGE_KEY)
    s = create_client(SUPABASE_URL, SUPABASE_KEY)
    return c, v, s

claude, voyage, supabase = init_clients()

# ── Search ────────────────────────────────────────────────────────────────────

def search_clauses(query: str, authority: str = None, top_k: int = 6):
    results = []
    clause_match = re.search(r"\d+\.\d+[\.\d]*", query)
    if clause_match:
        q = supabase.table("compliance_clauses").select("*")
        if authority and authority != "All":
            q = q.eq("authority", authority)
        results.extend(q.ilike("clause_number", f"%{clause_match.group()}%").execute().data)

    try:
        emb = voyage.embed([query], model="voyage-3").embeddings[0]
        params = {
            "query_embedding":  emb,
            "match_count":      top_k,
            "filter_authority": authority if authority and authority != "All" else None
        }
        results.extend(supabase.rpc("match_compliance_vectors", params).execute().data)
    except Exception as e:
        st.warning(f"Vector search error: {e}")

    seen, unique = set(), []
    for r in results:
        k = r.get("clause_number")
        if k not in seen:
            seen.add(k)
            unique.append(r)
    return unique[:top_k]

# ── Answer ────────────────────────────────────────────────────────────────────

def get_answer(question: str, clauses: list):
    if not clauses:
        return "No relevant clauses found in the database. Please check that the authority code has been loaded."

    context = ""
    for i, c in enumerate(clauses, 1):
        context += f"\n[{i}] {c.get('authority','BCA')} Clause {c.get('clause_number','')} — {c.get('topic','')}\n"
        context += f"Requirement: {c.get('requirement', c.get('text',''))}\n"

    msg = f"<authority_codes>\n{context}\n</authority_codes>\n\nQuestion: {question}"

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": msg}]
    )
    return response.content[0].text

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SG Compliance Assistant", page_icon="🏗️", layout="wide")
st.title("🏗️ Singapore Compliance Assistant")
st.caption("BCA · URA · SCDF · PUB · NEA · LTA · NParks")

with st.sidebar:
    st.header("Settings")
    authority = st.selectbox("Filter by authority",
                             ["All", "BCA", "URA", "SCDF", "PUB", "NEA", "LTA", "NParks"])
    st.divider()
    st.markdown("**Loaded codes:**")
    st.success("✓ BCA Accessibility 2025")
    st.info("More codes coming soon")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I can answer compliance questions based on Singapore authority codes. Try asking about corridor widths, ramp gradients, accessible toilets, or any BCA accessibility requirement."
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "clauses" in msg:
            with st.expander(f"📋 {len(msg['clauses'])} clauses retrieved"):
                for c in msg["clauses"]:
                    st.markdown(f"**{c.get('authority','BCA')} §{c.get('clause_number','')}** — {c.get('topic','')}")
                    req = c.get('requirement', c.get('text',''))
                    st.caption(req[:200] + "..." if len(req) > 200 else req)
                    st.divider()

if question := st.chat_input("Ask a compliance question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching clauses and reasoning..."):
            clauses = search_clauses(question, authority)
            answer  = get_answer(question, clauses)
        st.markdown(answer)
        if clauses:
            with st.expander(f"📋 {len(clauses)} clauses retrieved"):
                for c in clauses:
                    st.markdown(f"**{c.get('authority','BCA')} §{c.get('clause_number','')}** — {c.get('topic','')}")
                    req = c.get('requirement', c.get('text',''))
                    st.caption(req[:200] + "..." if len(req) > 200 else req)
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "clauses": clauses
    })
