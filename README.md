# arxiv-paper-agent

Fetch latest papers from **arXiv** by query, analyze & export, and summarize each paper in English (3 bullets: **What / Novelty / Relevance**) using **Groq (Llama 3.3 70B)**. Scholar results are added as **links only** via DuckDuckGo.

## 🔗 One-click (Colab)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TANGBUDU/arxiv-paper-agent/blob/HEAD/notebooks/fetch_analyze_arxiv.ipynb)

## 📦 Setup (local)
```bash
conda create -n arxiv-agent python=3.11 -y
conda activate arxiv-agent
pip install -r requirements.txt
