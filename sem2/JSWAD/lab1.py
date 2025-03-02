import random


def factorial(input_number: int) -> int:
    if input_number == 0:
        return 1
    else:
        return input_number * factorial(input_number - 1)


def too_less_much_game(input_number: int):
    number = random.randint(1, 100)
    while input_number != number:
        if input_number < number:
            print("Za mało")
        else:
            print("Za dużo")
        input_number = int(input("Podaj liczbę: "))
    print("Zgadłeś!")


def quadratic_equation(a: int, b: int, c: int):
    delta = b**2 - 4 * a * c
    if delta < 0:
        return None
    elif delta == 0:
        x = -b / (2 * a)
        return x
    else:
        x1 = (-b - delta**0.5) / (2 * a)
        x2 = (-b + delta**0.5) / (2 * a)
        return x1, x2


def count_sum_of_number_between_range(a: int, b: int) -> int:
    if a > b:
        a, b = b, a
    return sum(range(a + 1, b))


def sum_number_untill_negative():
    print("Suma dodatnich liczb wproawdzonych przez użytkownika")
    sum = 0
    counter = 0
    while True:
        number = int(input("Podaj liczbę: "))
        if number < 0:
            break
        sum += number
        counter += 1
        print(f"Suma liczb: {sum}. Liczba wprowadzonych liczb: {counter}")

    mean = sum / counter
    print(f"Średnia arytmetyczna wprowadzonych liczb: {mean}")


def print_stars():
    input_number = int(input("Podaj liczbę wierszy do wyświetlenia: "))
    for i in range(input_number):
        print("*" * (i + 1))


def calculate_tax(income: float) -> float:
    untaxed_income = 5000
    first_tax_limit = 10000
    second_tax_limit = 20000
    if income < 0:
        return -1
    elif income <= untaxed_income:
        return 0
    elif income <= untaxed_income + first_tax_limit:
        return 0.1 * (income - untaxed_income)
    elif income <= untaxed_income + first_tax_limit + second_tax_limit:
        return 0.1 * first_tax_limit + 0.15 * (
            income - untaxed_income - first_tax_limit
        )
    else:
        return (
            0.1 * first_tax_limit
            + 0.15 * second_tax_limit
            + 0.2 * (income - untaxed_income - first_tax_limit - second_tax_limit)
        )


if __name__ == "__main__":
    print("JSwAD - Laboratorium 1 - Jan Banot")
    # 1. Silnia
    # input = 5
    # output = factorial(input)
    # print(output)

    # 2. Gra za dużo za mało
    # print("Gra za dużo za mało")
    # input_number = int(input("Podaj liczbę: "))
    # too_less_much_game(input_number)

    # 3. Obliczenia pierwiastków równania kwadratowego
    # f(x) = ax^2 + bx + c
    # a = 1
    # b = 2
    # c = 4
    # output = quadratic_equation(a, b, c)
    # if output:
    #     print(f"Pierwiastki równania kwadratowego: {output}")
    # else:
    #     print("Brak rzeczywistych pierwiastków równania kwadratowego")

    # 4. Suma liczb całkowitych pomiędzy podanymi liczbami
    # print("Suma liczb całkowitych pomiędzy podanymi liczbami")
    # a = int(input("Podaj pierwszą liczbę: "))
    # b = int(input("Podaj drugą liczbę: "))
    # output = count_sum_of_number_between_range(a, b)
    # print(f"Suma liczb całkowitych pomiędzy {a} i {b} wynosi: {output}")

    # 5. Suma dodatnich liczb
    # sum_number_untill_negative()

    # 6. Wyświetlanie gwiazdek
    # print_stars()

    # 7. Obliczanie podatku w Królestwie Naturladnii
    # income = float(input("Podaj swój dochód: "))
    # tax = calculate_tax(income)
    # if tax == -1:
    #     print("Podano nieprawidłowy dochód (mniejszy od 0)")
    # else:
    #     print(f"Podatek do zapłacenia: {tax}")
