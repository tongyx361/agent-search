import json
import os
import traceback
from typing import Optional

import regex
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io

# %% ../nbs/00_core.ipynb 0
N_KAGGLE_CPU = 4
TOOL_CONFIG_ID2DICT = {
    "deepseekmath": {
        "template_id": "python",
        "input_begin": "```python",
        "input_end": "```",
        "output_begin": "```output",
        "output_end": "```",
        # "trunc_len": None,
        "trunc_len": (50, 50),
        "elipsis": "...",
    }
}


class CodeCellTemplate:
    """Code cell template, corresponding to the LLM training data.

    Parameters
    ----------
    template_id : str, default: "python"
    input_begin : str, default: "```python"
    input_end : str, default: "```"
    output_code_prefix : str, default: "print("
        Prefix of code that will be executed to display the output.
    output_begin : str, default: "```output"
    output_end : str, default: "```"
    trunc_len : tuple[int, int], default: None
        The maximum lengths to truncate the output into the beginning and end.\n`None` / any negative / double non-positive values like `(0, 0)` mean no truncation.
    elipsis : str, default: "..."
        The elipsis to use when truncating the output.
    """

    def __init__(
        self,
        template_id: str = "deepseekmath",
        input_begin: str = "```python",
        input_end: str = "```",
        output_code_prefix: str = "print(",
        output_begin: str = "```output",
        output_end: str = "```",
        trunc_len: Optional[tuple[int, int]] = None,
        elipsis: str = "...",
    ):
        self.template_id = template_id
        self.input_begin = input_begin
        self.input_end = input_end
        self.output_code_prefix = output_code_prefix
        self.output_begin = output_begin
        self.output_end = output_end
        self.trunc_len = trunc_len
        self.elipsis = elipsis

    @staticmethod
    def load_from_id_or_path(tool_config: str = "deepseekmath") -> "CodeCellTemplate":
        """Load the configuration from the ID or path.

        Parameters
        ----------
        tool_config : str, default: "deepseekmath"
            ID / Path to file of the code executeion configuration.

        Returns
        -------
        CodeCellTemplate
            The code execution configuration object.
        """
        if tool_config in TOOL_CONFIG_ID2DICT:
            return CodeCellTemplate(**TOOL_CONFIG_ID2DICT[tool_config])
        elif isinstance(tool_config, str) and os.path.exists(tool_config):
            with open(tool_config, "r") as f:
                tool_config = json.loads(f.read())
            return CodeCellTemplate(**tool_config)
        else:
            return CodeCellTemplate()  # Default: "deepseekmath"

    def input_cnt(self, ctx: str) -> int:
        """The number of input code cells."""
        return ctx.count(self.input_begin)

    def output_cnt(self, ctx: str) -> int:
        """The number of code output cells."""
        return ctx.count(self.output_begin)

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
        input_cnt = self.input_cnt(ctx)
        output_cnt = self.output_cnt(ctx)
        if input_cnt == output_cnt:
            return 1
        elif output_cnt > input_cnt:
            return 2  # Must be an error
        else:
            return 0  # Normal

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
        cells = [
            snippet.split(self.input_end, 1)[0].strip()
            for snippet in text.split(self.input_begin)
            if self.input_end in snippet
        ]
        cells = [cell for cell in cells if cell]  # Remove empty cells
        return cells

    @property
    def do_trunc(self) -> bool:
        """Whether to truncate the output or not."""
        return (
            isinstance(self.trunc_len, tuple)
            and any(leng > 0 for leng in self.trunc_len)
            and all(leng >= 0 for leng in self.trunc_len)
        )

    def trunc_output(self, output: str) -> str:
        """Truncate the output to the maximum length if necessary.

        Parameters
        ----------
        output : str
            The output to truncate.

        Returns
        -------
        str
            The truncated output.
        """
        if self.do_trunc and len(output) > sum(self.trunc_len):
            len_begin, len_end = self.trunc_len
            return output[:len_begin] + self.elipsis + output[-len_end:]
        else:
            return output

    def wrap_output(self, output: str) -> str:
        """Return `f"\\n{self.output_begin}\\n{output}\\n{self.output_end}\\n"`
        following https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/infer/run_tool_integrated_eval.py#L182-L183
        """
        return (
            "\n"  # Assure the output is separated from the input
            + f"{self.output_begin}\n{output}\n{self.output_end}"
            + "\n"
        )  # Prevent LLM from trying to generate the output again

    def clean_stderr(self, stderr: str) -> str:
        """Clean the stderr output."""
        err_lines = []
        for line in stderr.split("\n"):  # Remove delimiter
            patt = regex.search(
                r'(?P<start>.*)File "(?P<file>.*)", line (?P<lno>\d+), (?P<end>.*)',
                line,
            )
            if patt is not None:
                if "<module>" in patt.group("end"):
                    continue
                fname = patt.group("file")
                if "site-packages" in fname:
                    fname = f"site-packages{fname.split('site-packages', 1)[1]}"
                    line = f'{patt.group("start")}File "{fname}", {patt.group("end")}'
                else:
                    line = f'{patt.group("start")}{patt.group("end")}'
            else:
                patt = regex.search(
                    r"(?P<start>.*)(?P<file>/.*site-packages/.*\.py)(?P<end>.*)", line
                )
                if patt is not None:
                    line = f'{patt.group("start")}site-packages{patt.group("file").split("site-packages", 1)[1]}{patt.group("end")}'
            err_lines.append(line)
        cleaned_stderr = "\n".join(err_lines)
        return cleaned_stderr


ANSI_ESC = regex.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def exec_cells(cells: list[str], color: bool = False) -> tuple[str, str]:
    """Execute the code cells like a notebook and return the stdout and stderr of the last cell.
    Modified from
    - https://github.com/Kipok/NeMo-Skills/blob/6a909ec0974340b02a1083dce90e79bea30ecb60/nemo_skills/code_execution/sandbox.py#L168-L233
    - https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/infer/run_tool_integrated_eval.py#L163-L180

    Parameters
    ----------
    cells : list[str]
        The code cells to execute.

    Returns
    -------

    """

    try:
        # Create a new InteractiveShell instance
        shell = InteractiveShell.instance()

        if not color:
            # Disable coloring
            shell.colors = "NoColor"
            shell.color_info = False
        shell.run_cell(
            """
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OPENBLAS_NUM_THREADS'] = '16'
from sympy import *
"""
        )
        for cell in cells:
            with io.capture_output(display=False) as captured:
                shell.run_cell(cell)
                # serializing to str to make sure things like Rational can be converted to json
                if not color:
                    output = ANSI_ESC.sub("", captured.stdout).strip()
                # assert not captured.stderr  # Always empty
    except Exception:
        # removing useless prefix from traceback
        output = traceback.format_exc().strip()

    return output
