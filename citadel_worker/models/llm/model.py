__all__ = ["LLMModel", "Prompter"]

import os
os.environ["USE_LIBUV"] = "0"
os.environ["VLLM_USE_V1"] = "0"
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.engine.arg_utils import  EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.executor.uniproc_executor import UniProcExecutor
import asyncio
from typing import Any, List, AsyncGenerator, Union
from citadel_shared.citadel_model import CitadelModel
from vllm.lora.request import LoRARequest
import time
import asyncio


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self):        
        self.template = {
            "description": "Template used for LLM.",
            "prompt_no_input": "<|start_header_id|>user<|end_header_id|>\n{instruction}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n",
            "response_split": "<|end_header_id|>"    
        }


    def generate_prompt(
        self,
        instruction: str,
        label: Union[None, str] = None,
    ) -> str:
        
        res = self.template["prompt_no_input"].format(
            instruction=instruction
        )
        if label:
            res = f"{res}{label}"
        
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

class LLMModel(CitadelModel):

    def init(self):
        #raise NotImplementedError("LLMModel is not implemented")
        model_path = self.config["model_path"]
        gpu_memory = self.config["gpu_memory"]

        engine_args = EngineArgs(
            model=model_path,
            tokenizer=model_path,
            dtype="half",
            max_model_len=8192,
            gpu_memory_utilization=gpu_memory,
            enable_lora=True
        )
        self.engine   = AsyncLLMEngine(engine_args.create_engine_config(),
                                       log_stats=True,
                                       executor_class=UniProcExecutor)
        self.prompter = Prompter()
        self.logger.info("LLMModel initialised (path=%s)", model_path)

    def finalize(self) -> None:
        self.engine.shutdown()

    def execute(self, inp):
        async def _collector():
            parts = []
            async for part in self.stream_execute(inp):
                parts.append(part)
            return "".join(parts)

        return asyncio.run(_collector())

    def execute_batch(self, input_data: List[Any]) -> List[Any]:
        return [self.execute(i) for i in input_data]

    async def stream_execute(self, inp) -> AsyncGenerator[str, None]:
        d = self.coerce_to_dict(inp)
        print(d, flush=True)
        txt = d["text"]
        temperature = float(d.get("temperature", 0.7))
        max_tokens = int(d.get("max_tokens", 256))
        top_p = float(d.get("top_p", 1.0))
        top_k = int(d.get("top_k", 50))
        expert_path = d.get("expert_path", None)
        expert_name = d.get("expert_name", None)

        prompt = self.prompter.generate_prompt(txt)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=[128001, 128009, 128007],
        )

        request_id = random_uuid()
        if expert_path and expert_name:
            lora_request = LoRARequest(
                expert_path,
                1,
                expert_name
            )
            results_generator = self.engine.generate(prompt, sampling_params, request_id, lora_request=lora_request)
        else:
            results_generator = self.engine.generate(prompt, sampling_params, request_id)

        prev_text  = ""
        start_time = time.time()
        async for ro in results_generator:
            cur = ro.outputs[0].text
            delta = cur[len(prev_text) :]
            prev_text = cur
            if delta:
                yield delta

        self.logger.info("Generation done in %.2f s", time.time() - start_time)

    def finalize(self) -> None:
        self.engine.shutdown()