import argparse
import random
import string


def generate_password(length: int = 16) -> str:
    if length < 8:
        raise ValueError("Password length must be at least 8 characters")
    
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for _ in range(length))
    return password


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a random password.")
    parser.add_argument('--length', type=int, default=16, help='Length of the password')
    
    args = parser.parse_args(argv)
    
    try:
        password = generate_password(args.length)
        print(password)
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    exit(main())
