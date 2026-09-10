# MvSED

Official project repository for **MvSED: Latency-Aware Selective Multi-View Retrieval for Unsupervised Social Event Clustering**.

[Project page](./index.html) · [Manuscript](./static/pdfs/MvSED.pdf) · [Research code](./code/) · [Reproduction guide](./code/README.md) · [Aggregate results](./code/results/paper_summary.csv)

MvSED treats multi-view retrieval as a bounded residual correction for streaming social event clustering. Text retrieval fixes the admissible candidate set, auxiliary views rerank only uncertain cases, and a frozen label-free gate accepts only supported repairs.

## Repository layout

```text
.
├── index.html                 # GitHub Pages project page
├── static/                    # Figures, styles, scripts, paper, code archive
├── code/                      # Research implementation and reproduction tools
│   ├── main.py
│   ├── data_process.py
│   ├── rerun_twitter18.py
│   ├── summarize_efficiency.py
│   ├── scripts/
│   ├── datasets/README.md
│   ├── SOURCE_PROVENANCE.md
│   └── results/paper_summary.csv
├── CITATION.cff
└── .gitignore
```

## Start with the code

The full setup, RAGFlow configuration, data placement, free anchor check, and controlled Event2012/Event2018 commands are documented in [`code/README.md`](./code/README.md).

```bash
cd code
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/anchor_smoke_test.sh
```

The smoke test is anchor-only and does not contact an LLM. Dataset files are not included; see [`code/datasets/README.md`](./code/datasets/README.md).

## Preview the project page locally

From the repository root:

```bash
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000`.

## Publish with GitHub Pages

1. Create a public GitHub repository, for example `MvSED`.
2. Upload the contents of this directory to the repository root.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, select **Deploy from a branch**.
5. Select `main` and `/ (root)`, then save.

The page will be published at `https://YOUR-USERNAME.github.io/MvSED/`. When served from GitHub Pages, the page automatically derives a link back to the repository's `code/` directory.

## Public-release boundary

This repository intentionally excludes:

- API keys and provider credentials;
- raw Twitter/Event2012/Event2018 message content;
- checkpoints, model weights, caches, and embedding files;
- full logs, prompts/responses, and message-level predictions.

Only aggregate paper values are included under `code/results/`. Rotate any credential that has previously appeared in a terminal, chat, screenshot, or shell history before making the repository public.

## Citation and acknowledgment

Citation metadata is provided in [`CITATION.cff`](./CITATION.cff). MvSED builds on the fixed-anchor detector and evolving event-memory protocol of RagSEDE; please cite the accompanying MvSED manuscript and the RagSEDE paper when using the inherited pipeline.

The project page is derived from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which was adopted from the Nerfies project page.

No software license is declared in this draft release. The authors should review the upstream RagSEDE license and select a compatible license before inviting third-party reuse.
