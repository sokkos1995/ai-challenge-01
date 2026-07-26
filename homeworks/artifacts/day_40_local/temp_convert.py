import sys


def c2f(celsius: float) -> float:
    return round((celsius * 9/5) + 32, 1)


def f2c(fahrenheit: float) -> float:
    return round((fahrenheit - 32) * 5/9, 1)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    if len(argv) != 3 or argv[1] not in ['c2f', 'f2c']:
        print("Usage: temp_convert.py <c2f|f2c> <temperature>")
        return 2

    try:
        temperature = float(argv[2])
    except ValueError:
        print("Invalid temperature")
        return 2

    if argv[1] == 'c2f':
        result = c2f(temperature)
        print(f"{temperature}°C is {result}°F")
    else:
        result = f2c(temperature)
        print(f"{temperature}°F is {result}°C")

    return 0


if __name__ == "__main__":
    sys.exit(main())
