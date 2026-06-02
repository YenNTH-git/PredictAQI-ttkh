import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_excel("tong_hop.xlsx")
df.columns = df.columns.str.strip()

# =========================
# 2. HÀM TÍNH AQI TỪ PM2.5 (US EPA)
# =========================
def calculate_aqi_pm25(pm25):
    if pd.isna(pm25):
        return np.nan

    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5,250.4, 201,  300),
        (250.5,350.4, 301,  400),
        (350.5,500.4, 401,  500)
    ]

    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= pm25 <= Chigh:
            aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (pm25 - Clow) + Ilow
            return round(aqi, 2)   # 🔹 làm tròn 2 chữ số

    return np.nan

# =========================
# 3. TÍNH AQI
# =========================
df["AQI"] = df["PM2.5"].apply(calculate_aqi_pm25)

# =========================
# 4. XUẤT FILE
# =========================
output_file = "tong_hop_with_AQI.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ Đã tạo file: {output_file}")
print(df.head())
