# Source provenance

This public package was assembled from the final Event2018 anchor-aligned source snapshot supplied for the MvSED paper on 2026-09-10.

Core hashes before documentation-only packaging:

```text
main.py          31503097a7aafcb42a111a3a0ed25ee29f44be94179dcf3c6a3dd0e98cac9009
data_process.py  f88eda3127f906ea3ee106fcd46bf681d4e3fb9b4f3c40d20b337b64093b78c3
```

`main.py` and `data_process.py` are unchanged in this package. Packaging changes were limited to documentation, safe wrapper scripts, dependency manifests, removal of a hard-coded Hugging Face mirror from helper modules, environment-based credentials in `evolution/get_chunks.py`, and raw-string cleanup in an optional legacy utility.

## Historical runner boundary

`frozen_final_eval.py` contains an older Event2012 checksum guard (`ef3df494...`) that does not match the released `main.py`. It is retained as protocol provenance and is not advertised as a runnable entry point. Do not edit that checksum to make the check pass: doing so would misrepresent two different source snapshots as identical.

For this release, use:

- `rerun_twitter18.py` / `scripts/run_event2018.sh` for the aligned Event2018 protocol;
- `scripts/run_event2012.sh` for the explicit final Event2012 configuration;
- `scripts/anchor_smoke_test.sh` for a free anchor-only validation.
