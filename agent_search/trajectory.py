import json
from typing import Optional, TextIO

from vllm import CompletionOutput, RequestOutput, SamplingParams


class VLLMPythonMathTrajectory:
    def __init__(
        self,
        prompt: str,
    ):
        self.prompt: str = prompt
        # To collect from RequestOutput
        self.prompt_tok_cnt: Optional[int] = None
        self.prompt_logprobs: list[float] = []
        # Update with searching
        self.out_texts: list[str] = []  # Text / Code outout
        self.out_tok_cnt: int = 0
        self.tot_tok_cnt: int = 0
        # To collect from CompletionOutput
        # self.sampling_params_list: list[SamplingParams] = []
        self.temperatures: list[float] = []
        self.completion_tok_cnts: list[int] = []
        self.completion_cum_logprobs: list[float] = []
        # self.completion_logprobs_list: list[list[float]] = []
        self.completion_finish_reasons: list[str] = []
        self.completion_stop_reasons: list[str] = []
        # Conclude
        self.finish_reason: Optional[str] = None
        self.ans: Optional[int] = None

    @property
    def ctx(self) -> str:
        return self.prompt + "".join(self.out_texts)

    def collect_from_req_output(self, req_output: RequestOutput) -> None:
        if self.prompt_tok_cnt is None:
            self.prompt_tok_cnt = len(req_output.prompt_token_ids)
            if req_output.prompt_logprobs is not None:
                for tok_id2logprob in req_output.prompt_logprobs:
                    # None if the corresponding sequence group doesn't require prompt logprob.
                    if tok_id2logprob is not None:
                        self.prompt_logprobs.append(
                            {k: v.__dict__ for k, v in tok_id2logprob.items()}
                        )
        self.tot_tok_cnt = len(req_output.prompt_token_ids)
        # TODO: Try collectting information about appended code and output?

    def collect_from_completion(
        self, completion: CompletionOutput, sampling_params: SamplingParams
    ) -> None:
        # Update
        self.out_texts.append(completion.text)
        new_output_tok_cnt = len(completion.token_ids)
        self.out_tok_cnt += new_output_tok_cnt
        self.tot_tok_cnt += new_output_tok_cnt
        # Collect from CompletionOutput
        # self.sampling_params_list.append(sampling_params)
        self.temperatures.append(sampling_params.temperature)
        self.completion_tok_cnts.append(len(completion.token_ids))
        self.completion_cum_logprobs.append(completion.cumulative_logprob)
        # This seems to cause more memory cost
        # completion_logprobs = []
        # for tok_id2logprob in completion.logprobs:
        #     completion_logprobs.append(
        #         {k: v.__dict__ for k, v in tok_id2logprob.items()}
        #     )
        # self.completion_logprobs_list.append(completion_logprobs)
        self.completion_finish_reasons.append(completion.finish_reason)
        self.completion_stop_reasons.append(completion.stop_reason)

    def to_dict(self) -> dict:
        return {
            "ans": self.ans,
            "finish_reason": self.finish_reason,
            "tot_tok_cnt": self.tot_tok_cnt,
            "output_tok_cnt": self.out_tok_cnt,
            "prompt_tok_cnt": self.prompt_tok_cnt,
            "prompt_logprobs": self.prompt_logprobs,
            "completion_finish_reasons": self.completion_finish_reasons,
            "completion_stop_reasons": self.completion_stop_reasons,
            "completion_tok_cnts": self.completion_tok_cnts,
            "completion_logprobs_list": self.completion_logprobs_list,
            "sampling_params_list": self.sampling_params_list,
            "out_texts": self.out_texts,
        }

    def dump(self, file: TextIO) -> None:
        if file:
            file.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
