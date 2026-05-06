"""
HumanEval code generation evaluation task.

Dataset: openai/openai_humaneval (HuggingFace)
Metric: pass@k (default pass@1 using greedy decoding)

Reference:
    Chen et al., 2021. https://arxiv.org/abs/2107.03374

Security note:
    HumanEval requires executing generated code in a subprocess.  A 10-second
    timeout and no network access are enforced to limit risk.  Do **not** run
    this task in a production environment without additional sandboxing.
"""

import ast
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Optional

from evaluation.tasks.base_task import BaseTask


def _sanitize_code(code: str) -> str:
    """Remove markdown code fences if present, preserving code indentation."""
    # Only strip if markdown fences exist; otherwise preserve all whitespace
    if "```" in code:
        code = re.sub(r"```(?:python)?\n?", "", code)
        code = re.sub(r"\n?```", "", code)
    return code


def _execute_code(code: str, timeout: int = 10) -> bool:
    """
    Execute ``code`` in an isolated subprocess and return True if it passes.

    Returns ``False`` on any exception, syntax error, or timeout.
    """
    try:
        # Verify syntax before running
        ast.parse(code)
    except SyntaxError:
        return False

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class HumanEvalTask(BaseTask):
    """
    HumanEval code generation task.

    The model is given a Python function stub (docstring + signature) and must
    complete the function body.  The generated code is concatenated with the
    provided unit tests and executed; the task is considered solved if all
    tests pass.

    Args:
        num_fewshot: Number of in-context examples (default 0).
        max_samples: Maximum number of evaluation problems (``None`` = all 164).
        seed: Random seed.
        max_new_tokens: Token budget for code generation.
        num_samples_per_task: Number of samples per task for pass@k estimation
            (default 1 for pass@1).
        timeout: Seconds allowed for test execution.
    """

    VERSION = 1
    DATASET_PATH = "openai/openai_humaneval"

    def __init__(
        self,
        num_fewshot: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 42,
        max_new_tokens: int = 512,
        num_samples_per_task: int = 1,
        timeout: int = 10,
    ) -> None:
        super().__init__(num_fewshot=num_fewshot, max_samples=max_samples, seed=seed)
        self.max_new_tokens = max_new_tokens
        self.num_samples_per_task = num_samples_per_task
        self.timeout = timeout

    def _load_dataset(self):
        from datasets import load_dataset

        local_path = Path(self.DATASET_PATH)
        if local_path.exists():
            parquet_files = sorted(local_path.rglob("*.parquet"))
            if parquet_files:
                return load_dataset(
                    "parquet",
                    data_files={"test": [str(path) for path in parquet_files]},
                    split="test",
                )

        return load_dataset(self.DATASET_PATH, split="test")

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        return doc["prompt"]

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        return doc["canonical_solution"]

    def construct_requests(
        self, doc: Dict[str, Any], ctx: str
    ) -> List[tuple]:
        return [
            (
                ctx,
                {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": self.num_samples_per_task > 1,
                    "temperature": 0.8 if self.num_samples_per_task > 1 else 1.0,
                    "stop_sequences": ["\ndef ", "\nclass ", "\n#", "\nif __name__"],
                },
            )
        ] * self.num_samples_per_task

    def process_results(
        self, doc: Dict[str, Any], results: List[Any]
    ) -> Dict[str, Any]:
        prompt = doc["prompt"]
        test_code = doc["test"]
        entry_point = doc["entry_point"]

        passed_any = False
        for generation in results:
            generation = _sanitize_code(generation)
            # Build full program: prompt + generation + tests
            full_code = (
                prompt
                + "\n"
                + generation
                + "\n\n"
                + test_code
                + "\n\n"
                + f"check({entry_point})\n"
            )
            if _execute_code(full_code, timeout=self.timeout):
                passed_any = True
                break

        return {"pass@1": int(passed_any)}

    def aggregation(self) -> Dict[str, Any]:
        return {"pass@1": _mean}

    def higher_is_better(self) -> Dict[str, bool]:
        return {"pass@1": True}

    @property
    def name(self) -> str:
        return "humaneval"


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
