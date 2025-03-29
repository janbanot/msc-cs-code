import random


class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False


class ComplexNumber:
    def __init__(self, re=0, i=0):
        self.re = re
        self.i = i

    def modul(self):
        return (self.re**2 + self.i**2) ** 0.5

    @staticmethod
    def add(x1, x2):
        return ComplexNumber(x1.re + x2.re, x1.i + x2.i)

    @staticmethod
    def multiply(x1, x2):
        return ComplexNumber(
            x1.re * x2.re - x1.i * x2.i,
            x1.re * x2.i + x1.i * x2.re,
        )


class Fraction:
    def __init__(self, numerator=0, denominator=1):
        self.numerator = numerator
        self.denominator = denominator

    def simplify(self):
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        common_divisor = gcd(self.numerator, self.denominator)
        self.numerator //= common_divisor
        self.denominator //= common_divisor

    @staticmethod
    def add(fraction1, fraction2):
        numerator = (
            fraction1.numerator * fraction2.denominator
            + fraction2.numerator * fraction1.denominator
        )
        denominator = fraction1.denominator * fraction2.denominator
        result = Fraction(numerator, denominator)
        result.simplify()
        return result

    @staticmethod
    def subtract(fraction1, fraction2):
        numerator = (
            fraction1.numerator * fraction2.denominator
            - fraction2.numerator * fraction1.denominator
        )
        denominator = fraction1.denominator * fraction2.denominator
        result = Fraction(numerator, denominator)
        result.simplify()
        return result

    @staticmethod
    def multiply(fraction1, fraction2):
        numerator = fraction1.numerator * fraction2.numerator
        denominator = fraction1.denominator * fraction2.denominator
        result = Fraction(numerator, denominator)
        result.simplify()
        return result

    @staticmethod
    def divide(fraction1, fraction2):
        if fraction2.numerator == 0:
            raise ValueError("Cannot divide by zero")
        numerator = fraction1.numerator * fraction2.denominator
        denominator = fraction1.denominator * fraction2.numerator
        result = Fraction(numerator, denominator)
        result.simplify()
        return result


def n_fibbonacci(n):
    if n <= 0:
        return None
    elif n == 1:
        return 0
    elif n == 2:
        return 1

    a, b = 0, 1
    for _ in range(2, n):
        a, b = b, a + b
    return b


def n_fibbonacci_recursive(n):
    if n <= 0:
        return None
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    return n_fibbonacci_recursive(n - 1) + n_fibbonacci_recursive(n - 2)


class Animal:
    def __init__(self, name, age, species):
        self.name = name
        self.age = age
        self.species = species

    def introduce_self(self):
        print(f"I'm a {self.species} named {self.name} and I'm {self.age} years old.")


class Dog(Animal):
    def __init__(self, name, age):
        super().__init__(name, age, "Dog")

    def introduce_self(self):
        print("Woof!")
        super().introduce_self()


class Bird(Animal):
    def __init__(self, name, age):
        super().__init__(name, age, "Bird")

    def introduce_self(self):
        print("Tweet!")
        super().introduce_self()


class HomeZoo:
    def __init__(self):
        self.animals = []

    def add_animal(self, animal):
        if isinstance(animal, Animal):
            self.animals.append(animal)
        else:
            raise TypeError("Only Animal instances can be added.")

    def introduce_all(self):
        for animal in self.animals:
            animal.introduce_self()


class Cat:
    def __init__(self, name, favorite_toys, disliked_toys):
        self.name = name

        common_toys = set(favorite_toys) & set(disliked_toys)
        if common_toys:
            raise ValueError(f"Toys {common_toys} cannot be both liked and disliked!")

        self.favorite_toys = favorite_toys.copy()
        self.disliked_toys = disliked_toys.copy()

    def greet(self):
        print(f"Hi! I'm {self.name}.")

    def talk_about_toys(self):
        print(
            f"I have {len(self.favorite_toys)} favorite toys: {', '.join(self.favorite_toys)}"
        )
        print(
            f"There are {len(self.disliked_toys)} toys I can't stand: {', '.join(self.disliked_toys)}"
        )

    def like_toy(self, toy):
        if toy in self.favorite_toys:
            print(f"I already like the '{toy}'!")
            return False
        elif toy in self.disliked_toys:
            print(f"I can't like the '{toy}' because I hate it!")
            return False
        else:
            self.favorite_toys.append(toy)
            print(f"I liked a new toy: '{toy}'!")
            return True

    def hate_toy(self, toy):
        if toy in self.disliked_toys:
            print(f"I already hate the '{toy}'!")
            return False

        if toy in self.favorite_toys:
            self.favorite_toys.remove(toy)
            self.disliked_toys.append(toy)
            print(f"I suddenly stopped liking '{toy}'! Now I hate it!")
            return True
        else:
            self.disliked_toys.append(toy)
            print(f"I hate the new toy: '{toy}'!")
            return True

    def change_mind(self, toy):
        if toy in self.favorite_toys:
            return self.hate_toy(toy)
        elif toy in self.disliked_toys:
            self.disliked_toys.remove(toy)
            self.favorite_toys.append(toy)
            print(f"I unexpectedly started liking '{toy}', though I hated it before!")
            return True
        else:
            print(f"I don't know the toy '{toy}', so I can't change my mind about it.")
            return False


def read_toys_from_file(filename):
    with open(filename, "r") as f:
        toys = [line.strip() for line in f if line.strip()]
    print(f"Read {len(toys)} toys from file '{filename}'.")
    return toys


def simulate_cat_walking(cat, all_toys, iterations=100):
    print(f"\n===== Rozpoczynamy symulację z {cat.name} na {iterations} kroków =====")

    encountered_toys = []

    for step in range(1, iterations + 1):
        print(f"\nKrok {step}: {cat.name} spaceruje po korytarzu...")

        toy = random.choice(all_toys)
        encountered_toys.append(toy)
        print(f"{cat.name} znajduje pudełko z zabawką '{toy}'!")

        if toy in cat.favorite_toys:
            if random.random() < 0.3:
                print(f"{cat.name} uważa, że czas znienawidzić zabawkę '{toy}'.")
                cat.hate_toy(toy)
            else:
                print(f"{cat.name} cieszy się, widząc zabawkę, którą już lubi.")
        elif toy in cat.disliked_toys:
            if random.random() < 0.15:
                print(f"{cat.name} zmienia zdanie na temat zabawki '{toy}'.")
                cat.change_mind(toy)
            else:
                print(f"{cat.name} jest niezadowolony, widząc zabawkę, której nie lubi.")
        else:
            if random.random() < 0.7:
                print(f"{cat.name} jest podekscytowany nową zabawką!")
                cat.like_toy(toy)
            else:
                print(f"{cat.name} nie lubi tej nowej zabawki.")
                cat.hate_toy(toy)

        if step % 10 == 0:
            print(f"\nPo {step} krokach:")
            cat.talk_about_toys()

    toy_counts = {}
    for toy in encountered_toys:
        toy_counts[toy] = toy_counts.get(toy, 0) + 1

    unique_toys = len(toy_counts)
    most_common = max(toy_counts.items(), key=lambda x: x[1]) if toy_counts else None

    print("\n===== Symulacja zakończona =====")
    print(
        f"{cat.name} znalazł {len(encountered_toys)} pudełek z {unique_toys} różnymi rodzajami zabawek."
    )
    if most_common:
        print(
            f"Najczęściej spotykaną zabawką była '{most_common[0]}', znaleziona {most_common[1]} razy."
        )

    print("\nStan końcowy:")
    cat.talk_about_toys()

    return encountered_toys


if __name__ == "__main__":
    print("JSwAD - Laboratorium 2 - Klasy i funkcje - Jan Banot")

    # 1. Konto bankowe
    # account = BankAccount("123456789")
    # print(f"Stan konta: {account.balance}")
    # account.deposit(1000)
    # print(f"Stan konta po wpłacie: {account.balance}")
    # account.withdraw(500)
    # print(f"Stan konta po wypłacie: {account.balance}")

    # 2. Liczba zespolona
    # x1 = ComplexNumber(3, 4)
    # x2 = ComplexNumber(1, 2)
    # print(f"Moduł x1: {x1.modul()}")
    # print(f"Moduł x2: {x2.modul()}")
    # suma = ComplexNumber.add(x1, x2)
    # print(f"Suma: {suma.re} + {suma.i}i")
    # iloczyn = ComplexNumber.multiply(x1, x2)
    # print(f"Iloczyn: {iloczyn.re} + {iloczyn.i}i")

    # 3. Ułamek
    # u1 = Fraction(1, 2)
    # u2 = Fraction(3, 4)
    # print(f"Ułamek 1: {u1.numerator}/{u1.denominator}")
    # print(f"Ułamek 2: {u2.numerator}/{u2.denominator}")
    # suma = Fraction.add(u1, u2)
    # print(f"Suma: {suma.numerator}/{suma.denominator}")
    # roznica = Fraction.subtract(u1, u2)
    # print(f"Różnica: {roznica.numerator}/{roznica.denominator}")
    # iloczyn = Fraction.multiply(u1, u2)
    # print(f"Iloczyn: {iloczyn.numerator}/{iloczyn.denominator}")
    # iloraz = Fraction.divide(u1, u2)
    # print(f"Iloraz: {iloraz.numerator}/{iloraz.denominator}")

    # 4. Ciąg Fibonacciego
    # import timeit
    # n = 40
    # start_time = timeit.default_timer()
    # fib = n_fibbonacci(n)
    # end_time = timeit.default_timer()

    # start_time_recursive = timeit.default_timer()
    # fib_recursive = n_fibbonacci_recursive(n)
    # end_time_recursive = timeit.default_timer()

    # print(f"Iteracyjnie ({n}): {fib}, czas: {end_time - start_time:.10f} s")
    # print(f"Rekurencyjnie ({n}): {fib_recursive}, czas: {end_time_recursive - start_time_recursive:.10f} s")

    # 5. Domowe zoo
    # azor = Dog("Azor", 5)
    # reksio = Dog("Reksio", 2)
    # dudek = Cat("Dudek", 3)

    # zoo = HomeZoo()
    # zoo.add_animal(azor)
    # zoo.add_animal(reksio)
    # zoo.add_animal(dudek)
    # zoo.introduce_all()

    # 6. Kot
    # mruczek = Cat("Mruczek", ["piłka", "mysz", "sznurek"], ["dzwonek", "laser"])
    # mruczek.greet()
    # mruczek.talk_about_toys()
    # mruczek.like_toy("drapak")
    # mruczek.like_toy("piłka")
    # mruczek.like_toy("dzwonek")
    # mruczek.hate_toy("mysz")
    # mruczek.change_mind("laser")
    # mruczek.talk_about_toys()

    # 7. Symulacja - kot i zabawki
    # toys_filename = "sem2/JSWAD/data/cat_toys.txt"
    # all_toys = read_toys_from_file(toys_filename)

    # cat = Cat(
    #     "Mruczek",
    #     favorite_toys=["piłka"],
    #     disliked_toys=["dzwonek"],
    # )

    # cat.greet()
    # cat.talk_about_toys()
    # cat.talk_about_toys()

    # iterations = 100
    # encountered_toys = simulate_cat_walking(cat, all_toys, iterations)

    # print("\n===== Statystyki zabawek =====")
    # all_encountered = set(encountered_toys)
    # liked_count = sum(1 for toy in all_encountered if toy in cat.favorite_toys)
    # disliked_count = sum(1 for toy in all_encountered if toy in cat.disliked_toys)

    # print(f"Z {len(all_encountered)} unikalnych zabawek napotkanych:")
    # print(f"- {cat.name} polubił {liked_count} zabawek")
    # print(f"- {cat.name} znienawidził {disliked_count} zabawek")
    # print(
    #     f"- {len(all_encountered) - liked_count - disliked_count} zabawek nie zostało napotkanych"
    # )
