#pip install requests
import requests
import time
import csv


# Endpoint API eBay do wyszukiwania
url = "https://api.ebay.com/buy/browse/v1/item_summary/search"

# Application Token
# w miejsce "token" wpisujemy ciąg znaków wygenerowany poprzez konto deweloperskie ebay
access_token = "token"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Content-Language": "en-US"
}

# Pobieranie wyników wyszukiwania za pomocą frazy kluczowej
query = "iPhone 13"
limit = 200
offset = 0
all_items = []

while True:
    params = {
        "q": query,
        "limit": limit,
        "offset": offset
    }

    response = requests.get("https://api.ebay.com/buy/browse/v1/item_summary/search", headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        items = data.get("itemSummaries", [])
        if not items:
            break

        all_items.extend(items)
        print(f"Pobrano {len(items)} wyników z offsetem {offset}")

        offset += limit
        time.sleep(1)

        if offset >= data.get("total", 0):
            break
    else:
        print("Błąd:", response.status_code, response.text)
        break

print(f"\nŁącznie pobrano {len(all_items)} wyników.")

# Zapis do pliku CSV

with open("ebay_wyniki.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # nagłówki
    writer.writerow(["Tytuł", "Cena", "Waluta", "Link"])

    # dane
    for item in all_items:
        title = item.get("title", "brak")
        price = item.get("price", {}).get("value", "brak")
        currency = item.get("price", {}).get("currency", "brak")
        url = item.get("itemWebUrl", "brak")
        writer.writerow([title, price, currency, url])

print("Dane zapisane do ebay_wyniki.csv")