from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Student:
    name: str
    grades: list[float] = field(default_factory=list)

    def average(self) -> float:
        return 0.0 if not self.grades else sum(self.grades) / len(self.grades)


if __name__ == "__main__":
    import json, sys

    if len(sys.argv) != 2:
        print('usage: student_grade.py \'{"name":"Ada","grades":[5,4,5]}\'', file=sys.stderr)
        raise SystemExit(2)

    data = json.loads(sys.argv[1])
    student = Student(**data)
    print(f"{student.name}: {student.average():.2f}")
