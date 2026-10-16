"""Решения задач по основам Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(character.lower() in "aeiou" for character in text)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return number.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Длины векторов должны совпадать")
    if len(predicted) == 0:
        raise ValueError("Векторы не должны быть пустыми")
    return sum((x - y) ** 2 for x, y in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power:
            if power == 1:
                factors.append(f"({divisor})")
            else:
                factors.append(f"({divisor}**{power})")
        if divisor == 2:
            divisor = 3
        else:
            divisor += 2
    if number > 1:
        factors.append(f"({number})")
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    level = 0
    used = 0
    while used < cube_count:
        level += 1
        used += level ** 2
    return level if used == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    half = (len(digits) - 1) // 2
    if half == 0:
        return True
    left_sum = sum(int(digit) for digit in digits[:half])
    right_sum = sum(int(digit) for digit in digits[-half:])
    return left_sum == right_sum
