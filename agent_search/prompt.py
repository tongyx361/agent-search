import json
import logging
import os
from typing import Optional

PROMPT_TEMPLATE_ID2DICT = {
    "deepseekmath-tool": {  # c.f. https://github.com/deepseek-ai/DeepSeek-Math/tree/main/evaluation#3-evaluation
        "template_id": "deepseekmath-tool",
        "sys_prompt": "",
        "query_prompt": "User:" + " ",
        "prompt_after_query": (
            "\n"
            + "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
            + "\n\n"
        ),
        "resp_prompt": "Assistant:" + " ",
        "prompt_before_resp": "",
        "delim": "<｜end▁of▁sentence｜>",
    },
}


class PromptTemplate:
    """Prompt template, corresponding to the LLM training data.
    The complete prompt is in the form `{sys_prompt}{eg_qa1}{delim}{eg_qa2}{delim}...{delim}{eg_qaN}{delim}{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}`.
    default: PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"]

    Parameters
    ----------
    template_id : str
        Short name as ID of the prompt template, like "deepseekmath-tool".
    sys_prompt : str
        System prompt as the beginning of the full prompt.
    query_prompt : str
        Simple prompt as delimiter between response and new query.
    prompt_after_query : str
        Prompt to append after the raw query, like "Let's think step by step.".
    resp_prompt : str
        Simple prompt as delimiter between query and response.
    delim : str
        Delimiter between query-response pairs.
    """

    def __init__(
        self,
        template_id: str = PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"]["template_id"],
        sys_prompt: str = PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"]["sys_prompt"],
        query_prompt: str = PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"][
            "query_prompt"
        ],
        prompt_after_query: str = PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"][
            "prompt_after_query"
        ],
        resp_prompt: str = PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"]["resp_prompt"],
        prompt_before_resp: str = PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"][
            "prompt_before_resp"
        ],
        delim: str = PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"]["delim"],
    ):

        self.template_id = template_id
        self.sys_prompt = sys_prompt
        self.query_prompt = query_prompt
        self.prompt_after_query = prompt_after_query
        self.resp_prompt = resp_prompt
        self.prompt_before_resp = prompt_before_resp
        self.delim = delim

    @staticmethod
    def load_from_id_or_path(
        prompt_template: str = "deepseekmath-tool",
    ) -> "PromptTemplate":
        """Load prompt template from ID or file path."""
        if prompt_template in PROMPT_TEMPLATE_ID2DICT:  # ID
            return PromptTemplate(**PROMPT_TEMPLATE_ID2DICT[prompt_template])
        elif isinstance(prompt_template, str) and os.path.exists(prompt_template):
            # File path
            stem = os.path.splitext(os.path.basename(prompt_template))[0]
            with open(prompt_template, "r") as f:
                prompt_template = json.load(f)
            if "template_id" not in prompt_template:
                prompt_template["template_id"] = stem
            return PromptTemplate(**prompt_template)
        else:  # Default
            logging.warning(
                "Unknown prompt template, using the default 'deepseekmath-tool'."
            )
            return PromptTemplate(**PROMPT_TEMPLATE_ID2DICT["deepseekmath-tool"])

    def make_prefix_prompt(self, query: str) -> str:
        """Make a prefix prompt of `{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}.rstrip()`.
        NOTE: `.rstrip()` is important for correct tokenization."""
        return f"{self.query_prompt}{query}{self.prompt_after_query}{self.resp_prompt}{self.prompt_before_resp}".rstrip()

    def make_qa_pair(self, query: str, response: str) -> str:
        """Make a QA pair of `{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}{response}`."""
        return self.make_prefix_prompt(query) + response

    def make_full_prompt(
        self, query: str, eg_qas: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """Make full prompt as input to the model.
        Format: f"{sys_prompt}{eg_qa1}{eg_qa2}...{eg_qaN}{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}".
        """
        eg_qa_strs = [self.make_qa_pair(q, a) for q, a in eg_qas] if eg_qas else []
        prefix_prompt = self.make_prefix_prompt(query)
        return self.sys_prompt + self.delim.join(eg_qa_strs + [prefix_prompt])

    def get_last_resp(self, ctx: str) -> str:
        """Get the last response from the context."""
        return ctx.split(self.resp_prompt)[-1]
