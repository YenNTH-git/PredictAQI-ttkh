# ===============================
# 1. IMPORT THƯ VIỆN
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ===============================
# 2. TẠO DỮ LIỆU AQI (PM2.5, PM10, NO2)
# ===============================

np.random.seed(42)
n = 2000

PM25 = np.random.uniform(5, 200, n)
PM10 = np.random.uniform(10, 300, n)
NO2  = np.random.uniform(5, 100, n)

AQI = (
    0.6 * PM25 +
    0.3 * PM10 +
    0.1 * NO2 +
    np.random.normal(0, 20, n)
)

df = pd.DataFrame({
    'PM2.5': PM25,
    'PM10': PM10,
    'NO2': NO2,
    'AQI': AQI
})


# ===============================
# 3. KIỂM TRA DỮ LIỆU
# ===============================

print(df.info())
print(df.describe())
print(df.isnull().sum())


# ===============================
# 4. TÁCH FEATURE & TARGET
# ===============================

X = df[['PM2.5', 'PM10', 'NO2']]
y = df['AQI']


# ===============================
# 5. CHIA TRAIN / TEST
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 6. POLYNOMIAL FEATURES (TƯƠNG TÁC)
# ===============================

poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)


# ===============================
# 7. CHUẨN HÓA DỮ LIỆU
# ===============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled  = scaler.transform(X_test_poly)


# ===============================
# 8. LINEAR REGRESSION TỰ CÀI ĐẶT
# ===============================

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def train(self, X, y, learning_rate=0.01, num_iters=2000):
        n_samples, n_features = X.shape
        y = y.values.reshape(-1, 1)

        self.w = np.random.randn(n_features, 1) * 0.01
        self.b = np.mean(y)

        loss_hist = []

        for i in range(num_iters):
            y_hat = np.dot(X, self.w) + self.b
            errors = y_hat - y

            loss = np.mean(errors ** 2)
            loss_hist.append(loss)

            dw = np.dot(X.T, errors) / n_samples
            db = np.mean(errors)

            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            if i % 200 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

        return loss_hist

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def evaluate(self, X_test, y_test):
        y_test = y_test.reshape(-1, 1)
        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, mae, r2, y_pred


# ===============================
# 9. HUẤN LUYỆN
# ===============================

model = LinearRegression()
loss_hist = model.train(
    X_train_scaled,
    y_train,
    learning_rate=0.01,
    num_iters=2000
)


# ===============================
# 10. ĐÁNH GIÁ
# ===============================

mse, mae, r2, y_pred = model.evaluate(
    X_test_scaled,
    y_test.values
)

print("MSE :", mse)
print("RMSE:", np.sqrt(mse))
print("MAE :", mae)
print("R2  :", r2)


# ===============================
# 11. ĐỒ THỊ LOSS
# ===============================

plt.figure(figsize=(8,5))
plt.plot(loss_hist)
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Quá trình hội tụ của Linear Regression")
plt.grid(True)
plt.show()


# ===============================
# 12. ĐỒ THỊ SO SÁNH AQI
# ===============================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("AQI thực tế")
plt.ylabel("AQI dự đoán")
plt.title("So sánh AQI thực tế và AQI dự đoán")
plt.grid(True)
plt.show()


# ===============================
# 13. HỆ SỐ MÔ HÌNH
# ===============================

print("Hệ số mô hình:")
for name, coef in zip(poly.get_feature_names_out(), model.w.flatten()):
    print(f"{name}: {coef:.4f}")
