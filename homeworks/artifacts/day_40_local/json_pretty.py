import json
import sys


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    try:
        if len(argv) > 1:
            with open(argv[1], 'r') as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)

        print(json.dumps(data, indent=2))
        return 0
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Invalid JSON: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
