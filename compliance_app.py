"""
Singapore Compliance Assistant
===============================
- Text chat with RAG compliance search
- Voice recording with pydub conversion
- File upload next to chat input
- Plan/screenshot analysis via Claude vision
"""

import streamlit as st
import anthropic
import voyageai
from supabase import create_client
import re
import base64
import fitz
import speech_recognition as sr
import io
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# ── Config - keys loaded from Streamlit Secrets ───────────────────────────

ANTHROPIC_KEY = st.secrets["ANTHROPIC_KEY"]
VOYAGE_KEY    = st.secrets["VOYAGE_KEY"]
SUPABASE_URL  = st.secrets["SUPABASE_URL"]
SUPABASE_KEY  = st.secrets["SUPABASE_KEY"]

# ── Clients ───────────────────────────────────────────────────────────────

@st.cache_resource
def init_clients():
    c = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    v = voyageai.Client(api_key=VOYAGE_KEY)
    s = create_client(SUPABASE_URL, SUPABASE_KEY)
    return c, v, s

claude, voyage, supabase = init_clients()

# ── Search ────────────────────────────────────────────────────────────────

def search_clauses(query, authority=None, top_k=6):
    results = []
    m = re.search(r"\d+\.\d+[\.\d]*", query)
    if m:
        q = supabase.table("compliance_clauses").select("*")
        if authority and authority != "All":
            q = q.eq("authority", authority)
        results.extend(q.ilike("clause_number", f"%{m.group()}%").execute().data)
    emb = voyage.embed([query], model="voyage-3").embeddings[0]
    params = {
        "query_embedding" : emb,
        "match_count"     : top_k,
        "filter_authority": authority if authority and authority != "All" else None
    }
    results.extend(supabase.rpc("match_compliance_vectors", params).execute().data)
    seen, unique = set(), []
    for r in results:
        k = r.get("clause_number")
        if k not in seen:
            seen.add(k)
            unique.append(r)
    return unique[:top_k]

# ── Answer ────────────────────────────────────────────────────────────────

SYSTEM = """You are a Singapore building compliance assistant for an architecture firm.
You have access to official Singapore authority codes: BCA, URA, SCDF, PUB, NEA, LTA, NParks.

Rules:
1. ONLY answer using the provided code excerpts. Never rely on memory for clause numbers.
2. Always cite exact clause reference e.g. "BCA Clause 4.2.1".
3. If authorities conflict, flag it. Most stringent requirement always applies.
4. If excerpts don't cover the question, say so clearly.
5. Format: Requirement -> Clause reference -> Conflicts or notes.
6. Be concise and practical for busy architects.
7. When analysing uploaded plans or images, identify dimensions and labels, check against codes."""

def get_answer(question, clauses, image_b64=None, image_type="image/png"):
    ctx = ""
    for i, c in enumerate(clauses, 1):
        ctx += f"\n[{i}] {c.get('authority','BCA')} Clause {c['clause_number']} - {c['topic']}\n"
        ctx += f"Applies to: {c.get('applies_to','All buildings')}\n"
        ctx += f"Requirement: {c['requirement']}\n"
    text = f"<authority_codes>\n{ctx}\n</authority_codes>\n\nQuestion: {question}"
    if image_b64:
        content = [
            {"type":"image","source":{"type":"base64","media_type":image_type,"data":image_b64}},
            {"type":"text","text":text}
        ]
    else:
        content = text
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role":"user","content":content}]
    )
    return resp.content[0].text

# ── Voice transcription ───────────────────────────────────────────────────

def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language="en-SG")
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        st.error(f"Voice error: {e}")
        return ""

# ── File helpers ──────────────────────────────────────────────────────────

def pdf_first_page_b64(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
    return base64.b64encode(pix.tobytes("png")).decode(), "image/png"

def img_to_b64(img_bytes, mime):
    return base64.b64encode(img_bytes).decode(), mime

# ── Page setup ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SG Compliance Assistant",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Singapore Compliance Assistant")
st.caption("BCA · URA · SCDF · PUB · NEA · LTA · NParks")

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    authority = st.selectbox("Filter by authority",
        ["All","BCA","URA","SCDF","PUB","NEA","LTA","NParks"])

    st.divider()
    st.markdown("**Loaded codes:**")
    st.success("✓ BCA Accessibility 2025")
    st.success("✓ LTA Vehicle Parking 2019")
    st.success("✓ BCA Approved Document v7.08")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello! I can answer compliance questions based on Singapore authority codes. "
            "Try asking about corridor widths, ramp gradients, accessible toilets, "
            "parking provision, or any BCA / LTA requirement."
        )
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=350, caption="Uploaded file")
        if msg.get("clauses"):
            with st.expander(f"{len(msg['clauses'])} clauses retrieved - click to view"):
                for c in msg["clauses"]:
                    st.markdown(f"**{c.get('authority','BCA')} §{c['clause_number']}** - *{c['topic']}*")
                    req = c["requirement"]
                    st.caption(req[:300] + "..." if len(req) > 300 else req)
                    st.divider()

# ── Input row ─────────────────────────────────────────────────────────────

st.markdown("---")

col_voice, col_upload, col_spacer = st.columns([1.2, 1.8, 3])

with col_voice:
    st.markdown("**Voice**")
    audio = mic_recorder(
        start_prompt="Start",
        stop_prompt="Stop",
        just_once=True,
        use_container_width=True,
        key="mic"
    )

with col_upload:
    st.markdown("**Upload plan / screenshot**")
    uploaded = st.file_uploader(
        "upload",
        type=["pdf","png","jpg","jpeg"],
        label_visibility="collapsed",
        key="file_upload"
    )
    if uploaded:
        st.caption(f"✔ {uploaded.name}")

voice_question = ""
if audio and audio.get("bytes"):
    with st.spinner("Transcribing voice..."):
        voice_question = transcribe_audio(audio["bytes"])
    if voice_question:
        st.info(f"Heard: \"{voice_question}\"")
    else:
        st.warning("Could not understand audio - please try again or type below.")

question = st.chat_input("Ask a compliance question...")
final_q  = question or voice_question or None

# ── Process ───────────────────────────────────────────────────────────────

if final_q:
    image_b64, image_type, image_bytes_display = None, "image/png", None

    if uploaded:
        raw = uploaded.read()
        if uploaded.type == "application/pdf":
            image_b64, image_type = pdf_first_page_b64(raw)
        else:
            image_b64, image_type = img_to_b64(raw, uploaded.type)
            image_bytes_display   = raw

    user_msg = {"role": "user", "content": final_q}
    if image_bytes_display:
        user_msg["image_bytes"] = image_bytes_display
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(final_q)
        if image_bytes_display:
            st.image(image_bytes_display, width=350, caption="Uploaded file")

    with st.chat_message("assistant"):
        with st.spinner("Searching clauses and reasoning..."):
            clauses = search_clauses(final_q, authority)
            answer  = get_answer(final_q, clauses, image_b64, image_type)
        st.markdown(answer)
        if clauses:
            with st.expander(f"{len(clauses)} clauses retrieved - click to view"):
                for c in clauses:
                    st.markdown(f"**{c.get('authority','BCA')} §{c['clause_number']}** - *{c['topic']}*")
                    req = c["requirement"]
                    st.caption(req[:300] + "..." if len(req) > 300 else req)
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "clauses": clauses
    })
    st.rerun()
