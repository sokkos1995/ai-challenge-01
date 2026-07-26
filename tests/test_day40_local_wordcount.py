import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict

import pytest

# Load wordcount module dynamically
wordcount_path = Path("homeworks/artifacts/day_40_local/wordcount.py")
spec = importlib.util.spec_from_file_location("wordcount", wordcount_path)
wordcount_module = importlib.util.module_from_spec(spec)
sys.modules["wordcount"] = wordcount_module
spec.loader.exec_module(wordcount_module)

# Define test cases
test_cases: Dict[str, Any] = {
    "": 0,
    "hello world": 2,
    "one two three four five": 5,
    "a b c d e f g h i j k l m n o p q r s t u v w x y z": 26,
}

# Define pytest fixture
@pytest.fixture(scope="module")
def wordcount():
    return wordcount_module

# Define test function
def test_wordcount(wordcount):
    for input_str, expected_count in test_cases.items():
        result = subprocess.run(
            ["python", "-c", f"import wordcount; print(wordcount.wordcount('{input_str}'))"],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == str(expected_count), f"Failed for input: {input_str}"
