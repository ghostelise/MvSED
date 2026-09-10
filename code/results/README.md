# Result artifacts

`paper_summary.csv` contains only aggregate values reported on the project page and in the current manuscript. Raw messages, prompt/response transcripts, model-provider credentials, full run logs, checkpoints, and message-level predictions are intentionally excluded from this public package.

Generated experiment reports should be written under `results/generated/`, which is ignored by Git because those reports may contain message-level information. Before releasing any audit artifact, inspect it for raw text, provider metadata, and secrets.

