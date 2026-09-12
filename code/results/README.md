# Result artifacts

`paper_summary.csv` contains aggregate values reported on the project page and in the current manuscript, together with aggregation scope, timing scope, block range, token availability, and the anchor-verification boundary. A `main` row can combine a quality aggregate and a paired efficiency aggregate with different scopes; the two scope columns make that explicit.

For Event2012 `main` rows, quality is the mean over seeds 42/52/62, while latency, calls, failures, and tokens are from the controlled seed-42 paired run. Event2018 `main` rows use the aligned seed-42 run. Ablation rows use one fixed run: Event2012 M7--M21 and Event2018 M1--M16. `Proc` excludes initialization; `E2E` includes it. Blank token cells mean the raw totals were not included in this release, not zero use.

Generated experiment reports should be written under `results/generated/`, which is ignored by Git because those reports may contain message-level information. Before releasing any audit artifact, inspect it for raw text, provider metadata, and secrets.
