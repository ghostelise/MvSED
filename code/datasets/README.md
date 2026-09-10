# Dataset placement

MvSED does not redistribute Twitter/Event2012 or Twitter/Event2018 message content. Obtain the datasets under their original terms, then generate the chronological JSON caches with:

```bash
python twitter_12_process.py
python twitter_18_process.py
```

The runners expect:

```text
datasets/cache/twitter12.json
datasets/cache/twitter18.json
```

Each cache is a chronological list of blocks. The first entry is the initialization block; evaluated blocks follow the original RagSEDE indexing convention. Message records must retain the fields produced by `twitter_12_process.py` or `twitter_18_process.py`, including `text`, `event_id`, and `created_at`.

Do not commit the generated JSON, NumPy source arrays, user identifiers, raw tweet text, or message-level predictions. The repository-level `.gitignore` excludes these artifacts by default.

