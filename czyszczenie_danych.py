import sqlite3

# Połącz z bazą danych
conn = sqlite3.connect("iphone.db")
cursor = conn.cursor()

# Usuń wszystkie rekordy z tabeli iphones
cursor.execute("DELETE FROM iphones;")

# (Opcjonalnie) zresetuj automatyczne ID – tylko jeśli chcesz zaczynać od 1
cursor.execute("DELETE FROM sqlite_sequence WHERE name='iphones';")

conn.commit()
conn.close()

print("Wszystkie dane z tabeli 'iphones' zostały usunięte.")
