import json
import math
import os
from collections import Counter
from copy import deepcopy
from typing import Any, Callable, Optional, Union

from pebble import ProcessPool
from tqdm import tqdm
from vllm import LLM, RequestOutput, SamplingParams

from .exec_code import N_KAGGLE_CPU, CodeCellTemplate, exec_cells
from .parse import extract_ans_is, extract_boxed
from .prompt import PromptTemplate
from .trajectory import VLLMPythonMathTrajectory

N_MAX_COMPLETION_PER_REQ = 32


class AgentSearchCfg:

    def __init__(
        self,
        sampling_params: SamplingParams,
        n_fin_ans: int = 8,
        n_init_sample: int = 64,
        n_max_sample: Optional[int] = None,
        n_init_gen: int = 2,
        init_temperature: Optional[float] = None,
        code_temperature: Optional[float] = None,
        n_max_tot_tok: Optional[int] = None,
        timeout: int = 5,
        n_max_call: Optional[int] = None,
        allow_timeout: bool = True,
        allow_err: bool = True,
        allow_empty_output: bool = True,
        spec_neg: bool = False,
    ):
        self.sampling_params = sampling_params
        # if self.sampling_params.prompt_logprobs in [None, 0]:
        #     self.sampling_params.prompt_logprobs = 1
        # if self.sampling_params.logprobs in [None]:
        #     self.sampling_params.logprobs = 0
        self.n_fin_ans = n_fin_ans
        self.n_init_sample = (
            n_init_sample if n_init_sample is not None else self.n_fin_ans
        )
        self.n_max_sample = n_max_sample
        self.n_init_gen = n_init_gen
        self.init_temperature = (
            init_temperature
            if init_temperature is not None
            else sampling_params.temperature
        )
        self.code_temperature = (
            code_temperature
            if code_temperature is not None
            else sampling_params.temperature
        )
        self.n_max_tot_tok = n_max_tot_tok
        self.timeout = timeout
        self.n_max_call = n_max_call
        self.allow_timeout = allow_timeout
        self.allow_err = allow_err
        self.allow_empty_output = allow_empty_output
        self.spec_neg = spec_neg


class CtxTemplate:

    def __init__(
        self,
        prompt_template: Union[PromptTemplate, str],
        code_cell_template: Union[CodeCellTemplate, str],
    ):
        self.prompt_template = (
            prompt_template
            if isinstance(prompt_template, PromptTemplate)
            else PromptTemplate.load_from_id_or_path(prompt_template)
        )
        self.code_cell_template = (
            code_cell_template
            if isinstance(code_cell_template, CodeCellTemplate)
            else CodeCellTemplate.load_from_id_or_path(code_cell_template)
        )

    def get_stop_texts(self, gen_code: bool = False) -> list[str]:
        stop_texts = [
            self.prompt_template.query_prompt,
            self.prompt_template.resp_prompt,
            self.code_cell_template.output_begin,
        ]
        if gen_code:
            stop_texts.append(self.code_cell_template.input_begin)
        return [s.strip() for s in stop_texts]

    def input_cnt(self, ctx: str) -> int:
        """The number of input code cells."""
        last_resp = self.prompt_template.get_last_resp(ctx)
        return self.code_cell_template.input_cnt(last_resp)

    def output_cnt(self, ctx: str) -> int:
        """The number of code output cells."""
        last_resp = self.prompt_template.get_last_resp(ctx)
        return self.code_cell_template.output_cnt(last_resp)

    def check_cell_num(self, ctx: str) -> int:
        """Check the code / output cell number condition.

        Parameters
        ----------
        ctx : str
            The whole context containing all the code cells.

        Returns
        -------
        int
            0: Normal (code cells are less than output cells, i.e. there are some code cells to execute)
            1: No code cells to execute
            2: Output cells are more than input cells
        """
        last_resp = self.prompt_template.get_last_resp(ctx)
        return self.code_cell_template.check_cell_num(last_resp)

    def extract_cells(self, text: str) -> list[str]:
        """Extract code cells from the text.

        Parameters
        ----------
        text : str
            The text to extract code cells from.

        Returns
        -------
        list[str]
            The extracted code cells.
        """
        last_resp = self.prompt_template.get_last_resp(text)
        return self.code_cell_template.extract_cells(last_resp)


class AgentBase:
    def __init__(self):
        pass

    def ensemble_ans(self, fin_trajs: list[VLLMPythonMathTrajectory]) -> int:
        """Ensemble answers from finished trajectories.

        Parameters
        ----------
        fin_trajs : list[VLLMPythonMathTrajectory]
            The finished trajectories.

        Returns
        -------
        int
            The ensembled answer (non-negative integer) modulo 1000.
        """
        # TODO: consider trajectory information for ensemble
        if len(fin_trajs) == 0:
            print("[WARNING] No finished trajectories!")
            return 0

        # if isinstance(fin_trajs[0], dict):
        #     answers = [traj["ans"] for traj in fin_trajs]
        # else:
        #     answers = [traj.ans for traj in fin_trajs]
        # ans_votes = Counter(answers).most_common()
        # fin_ans = ans_votes[0][0]
        # print(
        #     f"{sum(ans_vote[1] for ans_vote in ans_votes)} answers vote as: {ans_votes} -> {fin_ans}"
        # )

        from collections import defaultdict
        from math import exp

        ans2tok_avg_prob = defaultdict(list)
        for traj in fin_trajs:
            tok_avg_cum_logprob = (
                traj.completion_cum_logprobs[-1] / traj.completion_tok_cnts[-1]
            )
            tok_avg_prob = exp(tok_avg_cum_logprob)
            ans2tok_avg_prob[traj.ans].append(tok_avg_prob)
        print(f"{ans2tok_avg_prob=}")
        ans2agg_tok_avg_prob = {
            ans: sum(tok_avg_probs) for ans, tok_avg_probs in ans2tok_avg_prob.items()
        }
        sorted_ans2agg_tok_avg_prob = sorted(
            ans2agg_tok_avg_prob.items(), key=lambda x: x[1], reverse=True
        )
        fin_ans = sorted_ans2agg_tok_avg_prob[0][0]
        print(f"{sorted_ans2agg_tok_avg_prob=} -> {fin_ans=}")

        return fin_ans

    def extract_ans(self, resp: str) -> str:
        """Extract the answer from the response."""
        ans = extract_boxed(resp)
        if ans is None:
            ans = extract_ans_is(resp)
        return ans

    def eq(self, a: Any, b: Any) -> bool:
        """Check the equality of two answers."""
        return a % 1000 == b % 1000


class VLLMPythonMathAgent(AgentBase):
    """Corresponding to a vLLM model trained on specific data"""

    def __init__(
        self,
        llm: LLM,
        prompt_template: Union[PromptTemplate, str],
        search_cfg: AgentSearchCfg,
        code_cell_template: Union[CodeCellTemplate, str],
        rag_eg_qa_map: Optional[dict[str, str]] = None,
    ):
        AgentBase.__init__(self)

        self.llm = llm
        self.ctx_template = CtxTemplate(prompt_template, code_cell_template)
        self.search_cfg = search_cfg
        self.rag_eg_qa_map = rag_eg_qa_map if rag_eg_qa_map else {}
        # Sampling parameters
        stop_texts = self.ctx_template.get_stop_texts(
            gen_code=self.search_cfg.code_temperature
            != self.search_cfg.sampling_params.temperature
        )
        for s in stop_texts:
            if s not in self.search_cfg.sampling_params.stop:
                self.search_cfg.sampling_params.stop.append(s)

    def is_searching_finished(
        self,
        fin_trajs: list[VLLMPythonMathTrajectory],
        drop_trajs: list[VLLMPythonMathTrajectory],
        ref_ans: Optional[Any] = None,
    ) -> bool:
        if ref_ans is not None:
            return any(self.eq(traj.ans, ref_ans) for traj in fin_trajs)
        else:
            return len(fin_trajs) >= self.search_cfg.n_fin_ans or (
                isinstance(self.search_cfg.n_max_sample, int)
                and len(fin_trajs) + len(drop_trajs) >= self.search_cfg.n_max_sample > 0
            )

    def search(
        self,
        init_input: str,
        ref_ans: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        fin_trajs, drop_trajs = [], []

        eg_qas = self.rag_eg_qa_map.items()
        prompt = self.ctx_template.prompt_template.make_full_prompt(
            init_input, eg_qas=eg_qas
        )

        try:
            while not self.is_searching_finished(fin_trajs, drop_trajs, ref_ans):

                if len(fin_trajs) > 0:
                    print(f"Ans. so far: {self.ensemble_ans(fin_trajs)}")
                trajs = [
                    VLLMPythonMathTrajectory(prompt=prompt)
                    for _ in range(
                        math.ceil(
                            self.search_cfg.n_init_sample / N_MAX_COMPLETION_PER_REQ
                        )
                    )
                ]
                text_gen_cnt = 0
                while True:  # Loop with steps
                    # LLM completion
                    sampling_params = self.search_cfg.sampling_params.clone()
                    if text_gen_cnt < self.search_cfg.n_init_gen:
                        # NOTE: We should not change the configuration value because they are used across multiple searchings
                        if text_gen_cnt == 0:
                            sampling_params.n = sampling_params.best_of = min(
                                self.search_cfg.n_init_sample, N_MAX_COMPLETION_PER_REQ
                            )
                        sampling_params.temperature = self.search_cfg.init_temperature
                        text_gen_cnt += 1
                    print(
                        f"{len(trajs)=} * {sampling_params.n=} = {len(trajs) * sampling_params.n} traj.s"
                    )
                    req_outputs: list[RequestOutput] = self.llm.generate(
                        prompts=[traj.ctx for traj in trajs],
                        sampling_params=sampling_params,
                    )
                    new_trajs: list[VLLMPythonMathTrajectory] = []
                    code_trajs: list[VLLMPythonMathTrajectory] = []
                    for traj, req_output in zip(trajs, req_outputs):
                        traj.collect_from_req_output(req_output)
                        for completion in req_output.outputs:
                            # Completion must be stopped because:
                            # - Maximum output token length reached
                            # - Stop strings generated:
                            #   - Code output "```output"
                            #   - New turn of conversation
                            #   - Code prompt "```python" if the coding temperature is set different
                            new_traj = deepcopy(traj)
                            new_traj.collect_from_completion(
                                completion, sampling_params
                            )
                            code_prompt = (
                                self.ctx_template.code_cell_template.input_begin.strip()
                            )
                            if new_traj.completion_stop_reasons[-1] == code_prompt:
                                code_trajs.append(new_traj)
                            else:
                                new_traj.ans = self.extract_ans(new_traj.out_texts[-1])
                                new_trajs.append(new_traj)

                    # Generate code if necessary
                    if code_trajs:
                        for code_traj in code_trajs:
                            code_traj.out_texts[-1] += code_prompt
                        code_sampling_params = sampling_params.clone()
                        code_sampling_params.n = code_sampling_params.best_of = 1
                        code_sampling_params.temperature = (
                            self.search_cfg.code_temperature
                        )
                        code_req_outputs: list[RequestOutput] = self.llm.generate(
                            prompts=[traj.ctx for traj in code_trajs],
                            sampling_params=code_sampling_params,
                        )
                        for code_traj, code_req_output in zip(
                            code_trajs, code_req_outputs
                        ):
                            code_traj.collect_from_req_output(req_output)
                            code_traj.collect_from_completion(
                                code_req_output.outputs[0], code_sampling_params
                            )
                            # TODO: selected = self.select_by_single_completion(code_traj) # ?

                    new_trajs = new_trajs + code_trajs

                    print(
                        f"After completion,       # of trajectories: finished: {len(fin_trajs)} | dropped: {len(drop_trajs)} | continue: {len(new_trajs)}"
                    )
                    # Select from old new_trajs
                    trajs = new_trajs
                    new_trajs = []
                    for traj in trajs:
                        selected = self.select_by_single_completion(traj)
                        if selected is True:
                            fin_trajs.append(traj)
                        elif selected is False:
                            drop_trajs.append(traj)
                        else:  # `None` -> Collect for further selection
                            new_trajs.append(traj)
                    print(
                        f"After single selection, # of trajectories: finished: {len(fin_trajs)} | dropped: {len(drop_trajs)} | continue: {len(new_trajs)}"
                    )

                    new_trajs, new_drop_trajs = self.select_by_multi_completions(
                        new_trajs
                    )
                    drop_trajs.extend(new_drop_trajs)
                    print(
                        f"After multi-selection,  # of trajectories: finished: {len(fin_trajs)} | dropped: {len(drop_trajs)} | continue: {len(new_trajs)}"
                    )
                    if (
                        self.is_searching_finished(fin_trajs, drop_trajs, ref_ans)
                        or not new_trajs
                    ):
                        break

                    # Code execution
                    with ProcessPool(
                        max_workers=min(N_KAGGLE_CPU, len(new_trajs)), max_tasks=1024
                    ) as pool:
                        # All trajectories without code to execute are dropped
                        iterator = pool.map(
                            exec_cells,  # Worker should be imported from an external module to avoid pickling issues when using multiprocessing in notebooks
                            [
                                self.ctx_template.extract_cells(traj.ctx)
                                for traj in new_trajs
                            ],
                            timeout=self.search_cfg.timeout,
                        ).result()
                        pbar = tqdm(total=len(new_trajs), desc="Executing")
                        results = []
                        while True:
                            try:
                                result = next(iterator)
                                results.append(result)
                            except StopIteration:
                                break
                            except Exception as e:
                                results.append(e)
                            pbar.update(1)
                        pbar.close()

                    # Select from old new_trajs
                    trajs = new_trajs
                    new_trajs = []
                    for traj, exec_res in zip(trajs, results):
                        if not isinstance(
                            exec_res, str
                        ):  # errors, e.g. `asyncio.TimeoutError`
                            if self.search_cfg.allow_timeout:
                                output = str(exec_res)
                            else:
                                traj.finish_reason = "exec-timeout"
                                drop_trajs.append(traj)
                                continue
                        else:
                            output = exec_res
                        stdout = output.split(
                            "---------------------------------------------------------------------------"
                        )[0]
                        if "Traceback (most recent call last)\n" in output:
                            stderr = self.ctx_template.code_cell_template.clean_stderr(
                                output.split("Traceback (most recent call last)")[-1]
                            )
                        else:
                            stderr = ""

                        if not self.search_cfg.allow_err and stderr:
                            traj.finish_reason = "exec-error"
                            drop_trajs.append(traj)
                        if not self.search_cfg.allow_empty_output and not stdout:
                            traj.finish_reason = "empty-output"
                            drop_trajs.append(traj)
                        output = stderr if stderr else stdout

                        # Compose output
                        output = self.ctx_template.code_cell_template.trunc_output(
                            output
                        )
                        if stderr:
                            output = "Runtime errors: " + output
                        traj.out_texts.append(
                            self.ctx_template.code_cell_template.wrap_output(output)
                        )
                        new_trajs.append(traj)
                    print(
                        f"After execution, # of trajectories: finished: {len(fin_trajs)} | dropped: {len(drop_trajs)} | continue: {len(new_trajs)}"
                    )
                    trajs = new_trajs
        except BaseException as e:
            raise e
        finally:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "cfg-search.json"), "w") as f:
                    json.dump(  # TODO: dump more configurations
                        {
                            "init_input": init_input,
                            **{
                                k: v
                                for k, v in self.search_cfg.__dict__.items()
                                if k != "sampling_params"
                            },
                            "sampling_params": self.search_cfg.sampling_params.__repr__(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                with open(os.path.join(output_dir, "fin-trajs-search.json"), "w") as f:
                    json.dump(
                        sorted(
                            [traj.to_dict() for traj in fin_trajs],
                            key=lambda x: x["ans"],
                        ),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                with open(os.path.join(output_dir, "drop-trajs-search.json"), "w") as f:
                    json.dump(
                        sorted(
                            [traj.to_dict() for traj in drop_trajs],
                            key=lambda x: x["finish_reason"],
                        ),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

        return self.ensemble_ans(fin_trajs)

    def select_by_multi_execs(self, trajs: list[VLLMPythonMathTrajectory]):
        """Select trajectories by multiple executions combined and set `traj.finish_reason` if needed.

        Parameters
        ----------
        trajs : list[VLLMPythonMathTrajectory]
            The trajectories to select.

        Returns
        -------
        tuple[list[VLLMPythonMathTrajectory], list[VLLMPythonMathTrajectory]]
            The selected trajectories and the dropped trajectories.
        """
        keep_trajs, drop_trajs = [], []

        for traj in trajs:
            if traj.finish_reason is not None:
                drop_trajs.append(traj)
            else:
                keep_trajs.append(traj)

    def select_by_single_completion(
        self, traj: VLLMPythonMathTrajectory
    ) -> Union[bool, None]:
        """Decide whether to finish or drop a trajectory by the single trajectory just after completion and set `traj.finish_reason` if needed.
        1. If answer is found, the trajectory is definitely finished, with no need to further search (Answer should only be found from LLM completion)
        2. Drop trajectories that
          1. are out of the model capabilities, e.g. longer than the context length
          2. must have errors, e.g. more output cells than code cells

        `traj.finish_reason` is set to:
        1. "valid-answer": Answer is found and valid （non-negative integer)
        2. "invalid-answer": Answer is found but invalid (other forms)
        3. "ctx-len": Context length is too long
        4. "no-answer": No answer is found but reaching EoS

        Parameters
        ----------
        traj : VLLMPythonMathTrajectory
            The trajectory to decide.

        Returns
        -------
        Union[bool, None]
            True if the trajectory is finished, False if the trajectory is dropped, None if the trajectory is still in search.
        """
        # Answer is found
        if traj.ans is not None:
            try:
                traj.ans = float(eval(traj.ans))
                assert traj.ans.is_integer()  # Integer
                traj.ans = round(traj.ans)
                if traj.ans < 0:
                    assert self.search_cfg.spec_neg
                    traj.ans *= -1
                return True
            except Exception:
                # print(f"Drop trajectory")
                traj.finish_reason = "invalid-answer"
                return False
        # Model capabilities limitation
        # 4096 for DeepSeekMath-7B models
        n_max_tot_tok = self.llm.llm_engine.scheduler.scheduler_config.max_model_len
        if (
            self.search_cfg.n_max_tot_tok is not None
            and self.search_cfg.n_max_tot_tok < n_max_tot_tok
        ):
            n_max_tot_tok = self.search_cfg.n_max_tot_tok
        if (
            # Over-esitmation of the token length of code output to append
            traj.tot_tok_cnt
            + (
                sum(self.ctx_template.code_cell_template.trunc_len)
                if self.ctx_template.code_cell_template.do_trunc
                else 0
            )
            >= n_max_tot_tok
        ):
            traj.finish_reason = "ctx-len"
            return False
        if (
            self.ctx_template.check_cell_num(traj.ctx) != 0
        ):  # Output cells are no less than code cells, i.e. no code cells to execute
            # but no answer is found and not reaching the length limit -> New turn of conversation is tried to generate without getting an answer
            traj.finish_reason = "no-answer"
            return False
        if (
            self.search_cfg.n_max_call is not None
            and self.ctx_template.input_cnt(traj.ctx) > self.search_cfg.n_max_call
        ):
            traj.finish_reason = "max-call"
            return False
        # NOTE: Heuristics are better considered with all the trajectories in `select_by_multi_completions`
        # Continue searching
        return None

    def select_by_multi_completions(
        self,
        trajs: list[VLLMPythonMathTrajectory],
    ) -> tuple[list[VLLMPythonMathTrajectory], list[VLLMPythonMathTrajectory]]:
        """Select trajectories by multiple completions combined and set `traj.finish_reason` if needed.

        Parameters
        ----------
        trajs : list[VLLMPythonMathTrajectory]
            The trajectories to select.

        Returns
        -------
        tuple[list[VLLMPythonMathTrajectory], list[VLLMPythonMathTrajectory]]
            The selected trajectories and the dropped trajectories.
        """
        keep_trajs, drop_trajs = [], []

        for traj in trajs:
            # TODO: Select considering multiple trajectories, e.g. the lowest perplexity / number of code cells
            keep_trajs.append(traj)

        return keep_trajs, drop_trajs
