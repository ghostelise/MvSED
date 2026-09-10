from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.dataset import DataSet
from ragflow_sdk.modules.chat import Chat
import os
import sys
import time
import json

HOST_ADDRESS = os.environ.get("RAGFLOW_HOST", "http://127.0.0.1:9380")
API_KEY = os.environ.get("RAGFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "RAGFLOW_API_KEY is not set. Export it in the current shell; "
        "do not store a real key in this file."
    )

# for twitter12
N = 22
DATASET = "twitter12"

# for twitter18
# N = 17
# DATASET = "twitter18"

# create a ragflow instance
ragflow_instance = RAGFlow(api_key=API_KEY, base_url=HOST_ADDRESS)
for b in range(1,N):
    chunk_list = []
    dataset = ragflow_instance.list_datasets(name = f"{DATASET}_SED_M{b}")[0]
    doc = dataset.list_documents(keywords=f"{DATASET}_SED_M{b}.txt")[0]
    i = 0
    for chunk in doc.list_chunks(page_size=900):
        i += 1
        chunk_list.append({'content':chunk.content, 'important_keywords':chunk.important_keywords})

    with open(f'evolution/{DATASET}/M{b}_chunk/M{b}.json', 'w', encoding='utf-8') as f:
        json.dump(chunk_list, f, ensure_ascii=False, indent=4)
