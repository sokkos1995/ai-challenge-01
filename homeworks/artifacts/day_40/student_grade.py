"""Student grades via dataclass (day40 refactor)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass


@dataclass
class Student:
    name: str
    grades: list[float]

    def average(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: student_grade.py \'{"name":"Ada","grades":[5,4,5]}\'', file=sys.stderr)
        raise SystemExit(2)
    payload = json.loads(sys.argv[1])
    student = Student(name=str(payload["name"]), grades=[float(g) for g in payload["grades"]])
    print(f"{student.name}: {student.average():.2f}")
