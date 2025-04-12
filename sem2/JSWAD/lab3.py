import pandas as pd
import random

print("JSwAD - Laboratorium 3 - Pandas - Jan Banot")

# Ćwiczenie 1
automobile = pd.read_csv("sem2/JSWAD/data/automobile_data.csv")
print(automobile.head())
print(automobile.tail())

# Ćwiczenie 2
automobile.replace(["?", "N.a"], pd.NA, inplace=True)
automobile.to_csv("sem2/JSWAD/data/automobile_data_cleaned.csv", index=False)

automobile = pd.read_csv("sem2/JSWAD/data/automobile_data_cleaned.csv")

# Ćwiczenie 3
automobile["price"] = pd.to_numeric(automobile["price"], errors="coerce")
most_expensive_car = automobile.loc[automobile["price"].idxmax()]
print(
    f"Najdroższa firma: {most_expensive_car['company']}, Cena: {most_expensive_car['price']}"
)

# Ćwiczenie 4
toyota_cars = automobile[automobile["company"] == "toyota"]
print("Szczegóły dotyczące samochodów Toyota:")
print(toyota_cars)

# Ćwiczenie 5
car_counts_by_company = automobile["company"].value_counts()
print("Liczba samochodów z podziałem na firmę:")
print(car_counts_by_company)

# Ćwiczenie 6
highest_priced_cars = automobile.loc[automobile.groupby("company")["price"].idxmax()]
print("Samochód o najwyższej cenie w każdej firmie:")
print(highest_priced_cars)

# Ćwiczenie 7
average_mileage_by_company = automobile.groupby("company")["average-mileage"].mean()
print("Średni przebieg dla każdej firmy:")
print(average_mileage_by_company)

# Ćwiczenie 8
sorted_automobiles = automobile.sort_values(by="price", ascending=True)
print("Samochody posortowane według ceny:")
print(sorted_automobiles)

# Ćwiczenie 9
GermanCars = {
    "Company": ["Ford", "Mercedes", "BMV", "Audi"],
    "Price": [23845, 171995, 135925, 71400],
}
japaneseCars = {
    "Company": ["Toyota", "Honda", "Nissan", "Mitsubishi"],
    "Price": [29995, 23600, 61500, 58900],
}

german_cars_df = pd.DataFrame(GermanCars)
japanese_cars_df = pd.DataFrame(japaneseCars)

combined_cars_df = pd.concat([german_cars_df, japanese_cars_df], ignore_index=True)
print("Połączone ramki danych:")
print(combined_cars_df)

# Ćwiczenie 10
Car_Price = {
    "Company": ["Toyota", "Honda", "BMV", "Audi"],
    "Price": [23845, 17995, 135925, 71400],
}
car_Horsepower = {
    "Company": ["Toyota", "Honda", "BMV", "Audi"],
    "horsepower": [141, 80, 182, 160],
}

car_price_df = pd.DataFrame(Car_Price)
car_horsepower_df = pd.DataFrame(car_Horsepower)

combined_cars_df = pd.merge(car_price_df, car_horsepower_df, on="Company")
print("Połączone ramki danych z nową kolumną:")
print(combined_cars_df)

# Ćwiczenie 11
world_alcohol = pd.read_csv("sem2/JSWAD/data/world_alcohol_data.csv")
random_row_count = random.randint(1, len(world_alcohol))
random_rows = world_alcohol.sample(n=random_row_count)

print(f"Wybrano losową liczbę wierszy: {random_row_count}")
print(random_rows)

# Ćwiczenie 12
# Pobranie roku od użytkownika
year = 1989
alcohol_consumption_by_year = world_alcohol[world_alcohol["Year"] == year]

if not alcohol_consumption_by_year.empty:
    print(f"Spożycie alkoholu w regionach w roku {year}:")
    print(alcohol_consumption_by_year)
else:
    print(f"Brak danych dla roku {year}.")

# Ćwiczenie 13
region = "Americas"
year = 1985

alcohol_consumption_america_1985 = world_alcohol[
    (world_alcohol["Year"] == year) & (world_alcohol["WHO region"] == region)
]

if not alcohol_consumption_america_1985.empty:
    print(f"Spożycie alkoholu w regionie '{region}' w roku {year}:")
    print(alcohol_consumption_america_1985)
else:
    print(f"Brak danych dla regionu '{region}' w roku {year}.")

# Ćwiczenie 14
average_consumption_threshold = 5
drink_type = "Beer"

filtered_records = world_alcohol[
    (world_alcohol["Display Value"] >= average_consumption_threshold)
    & (world_alcohol["Beverage Types"] == drink_type)
]

if not filtered_records.empty:
    print(
        f"Rekordy, w których średnie spożycie >= {average_consumption_threshold} i rodzaj napoju to '{drink_type}':"
    )
    print(filtered_records)
else:
    print(
        f"Brak rekordów, w których średnie spożycie >= {average_consumption_threshold} i rodzaj napoju to '{drink_type}'."
    )

# Ćwiczenie 15
average_consumption_threshold = 2
drink_type = "Wine"

filtered_wine_records = world_alcohol[
    (world_alcohol["Display Value"] > average_consumption_threshold)
    & (world_alcohol["Beverage Types"] == drink_type)
]

if not filtered_wine_records.empty:
    print(
        f"Rekordy, w których średnie spożycie > {average_consumption_threshold} i rodzaj napoju to '{drink_type}':"
    )
    print(filtered_wine_records)
else:
    print(
        f"Brak rekordów, w których średnie spożycie > {average_consumption_threshold} i rodzaj napoju to '{drink_type}'."
    )
