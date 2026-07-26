# Typing Cheatsheet for Python 3.8+

This cheatsheet provides a quick reference to common type annotations in Python 3.8+.

## Basic Types

- `int`: Integer
- `float`: Floating-point number
- `str`: String
- `bool`: Boolean (True/False)
- `NoneType`: None

## Container Types

- `list[T]`: List of elements of type T
- `tuple[T, ...]`: Tuple of elements of type T
- `set[T]`: Set of elements of type T
- `dict[K, V]`: Dictionary with keys of type K and values of type V

## Optional Types

- `Optional[T]`: Type or None (equivalent to Union[T, None])

## Union Types

- `Union[A, B, ...]`: Any of the types A, B, ...

## Callable Types

- `Callable[[A, B], C]`: Function that takes arguments of type A and B and returns a value of type C

## TypedDict
