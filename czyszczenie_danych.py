import sqlite3

# Połączenie z bazą danych
conn = sqlite3.connect("iphone.db")
cursor = conn.cursor()

# Usunięcie rekordów spełniających jeden z warunków:
# model = other
# zawiera frazę "for iphone" lub "for apple"
# zawiera frazę "case" lub "tempered glass" oraz cena jest niższa niż 200 USD
# cena jest wyższa niż 1599,99 USD
delete_query = """
DELETE FROM iphones
WHERE
    model = 'other'
    OR LOWER(title) LIKE '%for iphone%'
    OR LOWER(title) LIKE '%for apple%'
    OR (
        (LOWER(title) LIKE '%case%' OR LOWER(title) LIKE '%tempered glass%')
        AND price < 200
    )
    OR price > 1599.99;
"""

cursor.execute(delete_query)
conn.commit()

print(f"Usunięto {cursor.rowcount} rekordów.")

conn.close()
