import importlib.util
import os
from pathlib import Path
import pytest

# Load the fizzbuzz module dynamically
module_path = Path(__file__).parent / 'day_40_local' / 'fizzbuzz.py'
spec = importlib.util.spec_from_file_location("fizzbuzz", str(module_path))
fizzbuzz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fizzbuzz)

def test_fizzbuzz():
    assert fizzbuzz.fizzbuzz(3) == "Fizz"

def test_buzz():
    assert fizzbuzz.fizzbuzz(5) == "Buzz"

def test_fizzbuzz():
    assert fizzbuzz.fizzbuzz(15) == "FizzBuzz"
