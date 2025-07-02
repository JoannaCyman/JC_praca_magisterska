import sqlite3
import pandas as pd

# Funkcja do filtrowania tylko obserwacji w przedziale [Q1, Q3]
def keep_middle_iqr(group):
    Q1 = group['price'].quantile(0.25)
    Q3 = group['price'].quantile(0.75)
    return group[(group['price'] >= Q1) & (group['price'] <= Q3)]

# Połączenie z bazą danych
conn = sqlite3.connect("iphone.db")

# Wczytanie dane z bazy
df = pd.read_sql_query("SELECT * FROM iphones", conn)

# Zatrzymanie tylko dane pomiędzy Q1 a Q3 w każdej grupie
df_filtered = df.groupby(['model', 'condition', 'capacity'], group_keys=False).apply(keep_middle_iqr)

# Nadpisanie tabeli w bazie
df_filtered.to_sql("iphones", conn, if_exists="replace", index=False)
print("Dane w przedziale [Q1, Q3] zostały zapisane do bazy.")

conn.close()
