  """Мультиколлинеарность: VIF-анализ."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)
n = 200

# Сильно коррелированные признаки
x1 = np.random.normal(50, 10, n)
x2 = x1 + np.random.normal(0, 2, n)   # почти копия x1
x3 = np.random.normal(30, 5, n)        # независимый
x4 = x1 * 0.9 + np.random.normal(0, 3, n)  # коррелирует с x1

y = 2 * x1 + 3 * x3 + np.random.normal(0, 5, n)

df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})

# === 1. VIF ===
def calc_vif(data):
    vif = pd.DataFrame()
    vif["feature"] = data.columns
    vif["VIF"] = [variance_inflation_factor(data.values, i)
                  for i in range(data.shape[1])]
    return vif

print("=== VIF (все признаки) ===")
print(calc_vif(df))

# === 2. Удаляем признаки с VIF > 10 ===
while True:
    vif = calc_vif(df)
    max_vif = vif["VIF"].max()
    if max_vif < 10:
        break
    drop_col = vif.loc[vif["VIF"].idxmax(), "feature"]
    print(f"\nУдаляем {drop_col} (VIF = {max_vif:.2f})")
    df = df.drop(columns=[drop_col])

print("\n=== VIF (после удаления) ===")
print(calc_vif(df))

# === 3. Сравнение моделей ===
X_full = sm.add_constant(pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4}))
X_clean = sm.add_constant(df)

model_full = sm.OLS(y, X_full).fit()
model_clean = sm.OLS(y, X_clean).fit()

print(f"\nR² полной модели: {model_full.rsquared:.4f}")
print(f"R² очищенной:     {model_clean.rsquared:.4f}")
