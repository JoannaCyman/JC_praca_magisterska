import sqlite3

# Połączenie z bazą danych
conn = sqlite3.connect("iphone.db")
cursor = conn.cursor()

# Utworzenie tabeli
cursor.execute("""
CREATE TABLE IF NOT EXISTS iphones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    condition TEXT,
    capacity TEXT,
    price REAL,
    currency TEXT
);
""")
print("Tabela 'iphones' została utworzona.")

# Dodanie kolumn
def add_column_if_not_exists(column_name, column_type):
    try:
        cursor.execute(f"ALTER TABLE iphones ADD COLUMN {column_name} {column_type};")
        print(f"Kolumna '{column_name}' została dodana.")
    except sqlite3.OperationalError as e:
        if f"duplicate column name: {column_name}" in str(e).lower():
            print(f"Kolumna '{column_name}' już istnieje.")
        else:
            print("Błąd:", e)

add_column_if_not_exists("title", "TEXT")
add_column_if_not_exists("date_added", "TEXT")

# Zatwierdzenie zmian i zamknięcie połączenia
conn.commit()
conn.close()
