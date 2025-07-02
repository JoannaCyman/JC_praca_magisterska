# Usuwanie rekordów, które mają unknown zarówno w condition jak i capacity
import sqlite3

# Połączenie z bazą
conn = sqlite3.connect("iphone.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT COUNT(*) FROM iphones
    WHERE condition = 'unknown' AND capacity = 'unknown'
""")
print("Rekordów do usunięcia:", cursor.fetchone()[0])

# Usunięcie rekordów, które mają 'unknown' w obu kolumnach
cursor.execute("""
    DELETE FROM iphones
    WHERE condition = 'unknown' AND capacity = 'unknown'
""")

# Zatwierdzenie zmian i zamknięcie połączenia
conn.commit()
conn.close()

print("Rekordy zostały usunięte.")
