# app/streamlit_app.py
import os, re, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import streamlit as st
import arxiv

# ====================== 参数（默认更省） ======================
DEFAULT_MAX_RESULTS = 20   # 检索条数上限
DEFAULT_TOPK        = 5    # 只总结前 TOPK
DEFAULT_DAYS        = 365
DEFAULT_MIN_HITS    = 1    # 至少命中 1 个热词才算相关
DEFAULT_SLEEP       = 1.0  # 每条之间停顿，降低限流概率

TERMS = {
    "thermal","thermal image","thermal imaging","thermography","thermographic","thermogram",
    "infrared","infrared imaging","ir","ir imaging","flir","lwir","mwir","swir",
    "long-wave infrared","longwave infrared","radiometric","thermovision"
}

# ====================== 小工具 ======================
@st.cache_data(show_spinner=False, ttl=3600)
def build_query_from_free_text(user_q: str) -> str:
    q = (user_q or "").strip().replace('"','')
    toks = [t for t in re.split(r"\s+", q) if t]
    phrase = f'(ti:"{q}" OR abs:"{q}")'
    and_terms = [f"(ti:{t} OR abs:{t})" for t in toks if len(t)>2]
    return f"({phrase} OR ({' AND '.join(and_terms)}))" if and_terms else phrase

@st.cache_data(show_spinner=True, ttl=900)
def fetch_arxiv(query: str, max_results=DEFAULT_MAX_RESULTS, days_back=DEFAULT_DAYS) -> pd.DataFrame:
    search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=max_results)
    rows, seen = [], set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    for r in search.results():
        pub = r.published if r.published.tzinfo else r.published.replace(tzinfo=timezone.utc)
        if pub < cutoff: 
            continue
        key = (r.entry_id or "").lower()
        if key in seen: 
            continue
        seen.add(key)
        rows.append({
            "title": (r.title or "").strip(),
            "authors": ", ".join(a.name for a in r.authors),
            "summary": (r.summary or "").strip(),
            "published_utc": pub,
            "pdf_url": r.pdf_url,
            "arxiv_url": r.entry_id,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published_utc", ascending=False).reset_index(drop=True)
    return df

def relevance_hits(title, abstract):
    t = f"{title or ''} {abstract or ''}".lower()
    return sum(1 for k in TERMS if k in t)

# ====================== LLM（Groq） ======================
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, api_key: str):
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(model=model_name, temperature=0.2, max_tokens=384, request_timeout=60)

def call_llm_with_retry(llm, prompt: str, max_retries: int = 6):
    last = None
    for i in range(max_retries):
        try:
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ["RateLimit", "429", "Too Many Requests", "exceeded", "overloaded"]):
                wait = min(2**i, 20)  # 1,2,4,8,16,20...
                st.sidebar.write(f"⏳ Rate limited, retry {i+1}/{max_retries} after {wait}s")
                time.sleep(wait); last = e; continue
            last = e; break
    raise last

def summarize_one_en(llm, title: str, abstract: str, keyword: str):
    abs_short = (abstract or "")[:1600]  # 截断以省 token
    prompt = f"""Summarize in English using EXACTLY three one-sentence bullets.
- What it does:
- Novelty:
- Relevance to "{keyword}": (If not about the topic, output exactly: Not relevant to "{keyword}".)
Title: {title}
Abstract: {abs_short}
"""
    txt = call_llm_with_retry(llm, prompt)
    lines = [l.strip().lstrip("-*•").strip() for l in txt.splitlines() if l.strip()]
    out = []
    for l in lines:
        out.append(re.sub(r"^(What it does|Novelty|Relevance.*?):\s*", "", l, flags=re.I))
    while len(out) < 3:
        out.append("N/A")
    return out  # what, novelty, relevance

# ====================== UI ======================
st.set_page_config(page_title="arXiv Paper Agent", layout="wide")
st.title("arXiv Paper Fetcher & Analyzer")

with st.sidebar:
    st.subheader("Search & Controls")
    query     = st.text_input("Query", "thermal image")
    max_res   = st.slider("Max results (fetch from arXiv)", 10, 50, DEFAULT_MAX_RESULTS, step=5)
    topk      = st.slider("TOPK to summarize", 3, 20, DEFAULT_TOPK)
    days_back = st.select_slider("Time range (days)", options=[30, 90, 180, 365], value=DEFAULT_DAYS)
    min_hits  = st.slider("Min keyword hits (relevance)", 1, 3, DEFAULT_MIN_HITS)
    keep_only = st.checkbox("Keep only relevant in table", True)
    sleep_sec = st.select_slider("Sleep between LLM calls (sec)", options=[0.0, 0.5, 1.0, 1.5, 2.0], value=DEFAULT_SLEEP)
    model_name = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], index=1)  # 默认更省的 8B
    # API Key：优先用环境变量 / secrets；否则允许用户临时粘贴
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("GROQ_API_KEY (不会保存)", type="password")
    run = st.button("Run")

if not run:
    st.info("在左侧填好参数后点 **Run**。默认检索 20 条，只总结前 5 条相关论文。")
    st.stop()

if not api_key:
    st.error("缺少 GROQ_API_KEY。请在环境变量、.streamlit/secrets.toml 或侧边栏输入。")
    st.stop()

# 1) 检索
adv_q = build_query_from_free_text(query)
with st.spinner("Searching arXiv..."):
    df = fetch_arxiv(adv_q, max_results=max_res, days_back=days_back)

if df.empty:
    st.warning("No results.")
    st.stop()

# 2) 相关性打分与选择
df["relevance_hits"] = df.apply(lambda r: relevance_hits(r["title"], r["summary"]), axis=1)
df["relevant"] = df["relevance_hits"] >= min_hits
df = df.sort_values(["relevant", "published_utc"], ascending=[False, False]).reset_index(drop=True)

rows = df[df["relevant"]].head(topk)
if rows.empty:
    rows = df.head(topk)

# 3) 初始化 LLM
llm = get_llm(model_name, api_key)

# 4) 总结（仅对 rows，强节流）
data = []
pro = st.progress(0.0, text="Summarizing…")
for i, (_, r) in enumerate(rows.iterrows(), start=1):
    try:
        what, nov, rel = summarize_one_en(llm, r["title"], r["summary"], query)
        if not r["relevant"]:
            rel = f'Not relevant to "{query}"'
    except Exception as e:
        what = nov = "N/A"
        rel = f"Not summarized (error: {e.__class__.__name__})"
    data.append({
        "title": r["title"],
        "authors": r["authors"],
        "published_utc": r["published_utc"],
        "what": what, "novelty": nov, "relevance": rel,
        "pdf": r["pdf_url"], "arxiv": r["arxiv_url"]
    })
    pro.progress(i / len(rows))
    if sleep_sec:
        time.sleep(sleep_sec)

out_df = pd.DataFrame(data)
table = out_df if not keep_only else out_df[~out_df["relevance"].str.startswith("Not relevant")]
st.subheader("Summaries")
st.dataframe(table, use_container_width=True)

# 5) 允许下载
from io import StringIO, BytesIO
def to_markdown(df: pd.DataFrame) -> str:
    buf = []
    for _, r in df.iterrows():
        buf += [
            f"## {r['title']}",
            f"- Authors: {r['authors']}",
            f"- Date (UTC): {r['published_utc']:%Y-%m-%d}",
            f"- Links: [PDF]({r['pdf']}) | [arXiv]({r['arxiv']})",
            f"- **What**: {r['what']}",
            f"- **Novelty**: {r['novelty']}",
            f"- **Relevance**: {r['relevance']}",
            ""
        ]
    return "\n".join(buf)

st.download_button("Download CSV", data=table.to_csv(index=False).encode("utf-8-sig"), file_name="summary.csv", mime="text/csv")
st.download_button("Download Markdown", data=to_markdown(table), file_name="summary.md", mime="text/markdown")
