import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint

# 1. Wczytanie danych
conn = sqlite3.connect("iphone.db")
df = pd.read_sql_query("SELECT model, condition, capacity, price, date_added FROM iphones", conn)
conn.close()

# 2. Przetworzenie daty na zmienną numeryczną
df['date_added'] = pd.to_datetime(df['date_added'])
df['days_since_start'] = (df['date_added'] - df['date_added'].min()).dt.days

# 3. Usunięcie braków
df = df.dropna(subset=['model', 'condition', 'capacity', 'price', 'date_added'])

# 4. Konwersja zmiennych kategorycznych na typ "category"
for col in ['model', 'condition', 'capacity']:
    df[col] = df[col].astype('category')

# 5. Podział danych
X = df[['model', 'condition', 'capacity', 'days_since_start']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Optymalizacja hiperparametrów z RandomizedSearchCV
from xgboost import XGBRegressor
model = XGBRegressor(tree_method='hist', enable_categorical=True, objective='reg:squarederror', random_state=42)

param_distributions = {
    'max_depth': randint(2, 10),
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.2),
    'reg_alpha': uniform(0.0, 1.0),
    'reg_lambda': uniform(0.0, 1.0),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}

search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)
print("\nOptymalne parametry:")
print(search.best_params_)

# 7. Predykcja i ewaluacja
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.3f}")

# 8. Predykcja na konkretny dzień (08-05-2025) dla produktu o wyznaczonych parametrach (wprowadzone parametry są edytowalne)
target_date = pd.Timestamp("2025-05-08")
days_since_start = (target_date - df['date_added'].min()).days
example = pd.DataFrame([{
    'model': '13',
    'condition': 'good',
    'capacity': '128 GB',
    'days_since_start': days_since_start
}])

for col in ['model', 'condition', 'capacity']:
    example[col] = pd.Categorical(example[col], categories=df[col].cat.categories)

predicted_price = best_model.predict(example)
print(f"Przewidywana cena na 2025-05-08: {predicted_price[0]:.2f} USD")

# 9. Wykres błędów
plt.hist(y_test - y_pred, bins=30)
plt.title("Rozkład błędów predykcji")
plt.xlabel("Błąd")
plt.ylabel("Liczba obserwacji")
plt.show()

# 10. Wykres - wartości rzeczywiste vs. predykcja
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Cena rzeczywista")
plt.ylabel("Cena przewidywana")
plt.title("Cena rzeczywista vs. przewidywana")
plt.show()

# 11. Ważność cech
importances = best_model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Cecha': list(importances.keys()),
    'Wartość': list(importances.values())
}).sort_values(by='Wartość', ascending=False)

print("\nWażność cech wg XGBoost:")
print(importance_df.head(10).to_string(index=False))

importance_df.head(10).plot.bar(x='Cecha', y='Wartość', title='Ważność cech (weight)')
plt.tight_layout()
plt.show()

from xgboost import to_graphviz
import re
import graphviz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 12. Drzewo decyzyjne - wizualizacja
dot_source = to_graphviz(best_model, tree_idx=0, rankdir='LR').source

dot_source = re.sub(r'gain=([-+]?[0-9]*\.?[0-9]+)', lambda m: f"gain={int(round(float(m.group(1))))}", dot_source)
dot_source = re.sub(r'cover=([-+]?[0-9]*\.?[0-9]+)', lambda m: f"cover={int(round(float(m.group(1))))}", dot_source)
dot_source = re.sub(r'leaf=([-+]?[0-9]*\.?[0-9]+)', lambda m: f"leaf={float(m.group(1)):.1f}", dot_source)

graph = graphviz.Source(dot_source)
graph.format = 'png'
graph.render('simplified_tree', cleanup=True)

img = mpimg.imread('simplified_tree.png')
plt.figure(figsize=(20, 10))
plt.imshow(img)
plt.axis('off')
plt.title("Pierwsze drzewo decyzyjne XGBoost")
plt.show()

#13. Kody odpowiadające zmiennym
for col in ['model', 'condition', 'capacity']:
    print(f"\nZmienna: {col}")
    print(f"Cechy i ich kody numeryczne:")
    print(df[col].cat.codes.value_counts().sort_index())
    print(f"Cechy odpowiadające kodom:")
    print(dict(enumerate(df[col].cat.categories)))
