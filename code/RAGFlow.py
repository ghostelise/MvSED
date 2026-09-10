import time

from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.chat import Chat
from ragflow_sdk.modules.dataset import DataSet


class RAGFlowInstance:
    def __init__(
        self,
        HOST_ADDRESS,
        API_KEY,
        llm_model="deepseek-r1-distill-qwen-32b",
        top_p=0.3,
        max_tokens=512,
        ragflow_top_n=8,
        ragflow_top_k=1024,
        dataset_embedding_model="BAAI/bge-small-zh-v1.5",
        dataset_ready_attempts=60,
        dataset_ready_delay=10.0,
    ):
        self.ragflow_instance = RAGFlow(api_key=API_KEY, base_url=HOST_ADDRESS)
        self.llm_model = llm_model
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.ragflow_top_n = ragflow_top_n
        self.ragflow_top_k = ragflow_top_k
        self.dataset_embedding_model = dataset_embedding_model
        self.dataset_ready_attempts = dataset_ready_attempts
        self.dataset_ready_delay = dataset_ready_delay

    def creat_evaluator(self, name, evaluator_prompt, evaluator_presence_penalty, evaluator_frequency_penalty, evaluator_temperature):
        llm = Chat.LLM(self.ragflow_instance, {"model_name": self.llm_model,
                                               "temperature": evaluator_temperature,
                                               "top_p": self.top_p,
                                               "presence_penalty": evaluator_presence_penalty,
                                               "frequency_penalty": evaluator_frequency_penalty,
                                               "max_tokens": self.max_tokens, })
        
        prompt = Chat.Prompt(self.ragflow_instance, {"similarity_threshold": 0.2,
                                                     "keywords_similarity_weight": 0.3,
                                                     "top_n": self.ragflow_top_n,
                                                     "top_k": self.ragflow_top_k,
                                                     "variables": [{
                                                         "key": "knowledge",
                                                         "optional": True
                                                     }], 
                                                     "rerank_model": "",
                                                     "empty_response": None,
                                                     "opener": "Hi! I'm your assistant, what can I do for you?",
                                                     "show_quote": False,
                                                     "prompt": evaluator_prompt})
        
        try:
            self.evaluator = self.ragflow_instance.create_chat(name, llm=llm, prompt=prompt)
        except Exception as e:
            if "Duplicated chat name" in str(e):
                print(f"检测到同名对话 '{name}'，正在删除旧对话并重试...")
                try:
                     self.ragflow_instance.delete_chat(name)
                except Exception:
                     pass
                
                # 再次创建
                self.evaluator = self.ragflow_instance.create_chat(name, llm=llm, prompt=prompt)
            else:
                raise e
        
        return self.evaluator

    def creat_dataset(self, name):
        my_parser_config = DataSet.ParserConfig(self.ragflow_instance, {"chunk_token_num":128,"delimiter":"\\n!?;。；！？","html4excel":False,"layout_recognize":False,"raptor":{"user_raptor":False}})
        self.dataset_instance = self.ragflow_instance.create_dataset(
            name=name,
            embedding_model=self.dataset_embedding_model,
            parser_config=my_parser_config,
        )
        self.dataset_instance.upload_documents([{"display_name":name+'.txt',"blob":b'tmp'}])
        self.doc = self.dataset_instance.list_documents(keywords=name)[0]
        self.dataset_instance.async_parse_documents([self.doc.id])
        print("Creat KB...Async bulk parsing initiated.")
        for attempt in range(1, self.dataset_ready_attempts + 1):
            if self.doc.run == "DONE":
                break
            if self.doc.run in {"FAIL", "CANCEL"}:
                raise RuntimeError(
                    f"RAGFlow document parsing ended with state {self.doc.run}."
                )
            print(self.doc.run)
            time.sleep(self.dataset_ready_delay)
            self.doc = self.dataset_instance.list_documents(keywords=name)[0]
        else:
            raise TimeoutError(
                "RAGFlow document parsing timed out after "
                f"{self.dataset_ready_attempts} attempts."
            )
        for chunk in self.doc.list_chunks():
            self.doc.delete_chunks([chunk.id])
        event_list = ['Others']
        return self.doc, event_list        

    def creat_detector(
        self,
        name,
        detector_prompt,
        detector_similarity_threshold,
        detector_keywords_similarity_weight,
        detector_presence_penalty,
        detector_frequency_penalty,
        detector_temperature,
        attach_dataset=False,
    ):
        llm = Chat.LLM(self.ragflow_instance, {"model_name": self.llm_model,
                                               "temperature": detector_temperature,
                                               "top_p": self.top_p,
                                               "presence_penalty": detector_presence_penalty,
                                               "frequency_penalty": detector_frequency_penalty,
                                               "max_tokens": self.max_tokens, })
        
        prompt = Chat.Prompt(self.ragflow_instance, {"similarity_threshold": detector_similarity_threshold,
                                                     "keywords_similarity_weight": detector_keywords_similarity_weight,
                                                     "top_n": self.ragflow_top_n,
                                                     "top_k": self.ragflow_top_k,
                                                     "variables": [{
                                                         "key": "knowledge",
                                                         "optional": True
                                                     }], 
                                                     "rerank_model": "",
                                                     "empty_response": None,
                                                     "opener": "Hi! I'm your assistant, what can I do for you?",
                                                     "show_quote": False,
                                                     "prompt": detector_prompt})
        
        # Full and matched ablation variants receive explicit candidates from
        # main.py. The source-faithful RagSEDE variant sets attach_dataset=True
        # to reproduce RAGFlow's original attached-knowledge-base retrieval.
        create_kwargs = {"llm": llm, "prompt": prompt}
        if attach_dataset:
            create_kwargs["dataset_ids"] = [self.dataset_instance.id]

        try:
            self.detector = self.ragflow_instance.create_chat(
                name, **create_kwargs
            )
        except Exception as e:
             if "Duplicated chat name" in str(e):
                print(f"检测到同名 Detector '{name}'，正在重试...")
                try:
                     self.ragflow_instance.delete_chat(name)
                except Exception:
                     pass
                self.detector = self.ragflow_instance.create_chat(
                    name, **create_kwargs
                )
             else:
                raise e

        return self.detector
