
# ============================================================
# Taiwan House Sale Price Prediction
# Baseline Model: Random Forest Regression
# ============================================================

from pathlib import Path
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. 設定路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "train-v2.xlsx"
VALID_FILE = DATA_DIR / "valid-v2.xlsx"
TEST_FILE = DATA_DIR / "test-reindex-test-v2.2.xlsx"

# 自動尋找 sample submission
sample_files = list(DATA_DIR.glob("*sample*submission*.csv"))

if not sample_files:
    raise FileNotFoundError(
        "找不到 Sample Submission CSV，請放進 data 資料夾"
    )

SAMPLE_FILE = sample_files[0]


# ============================================================
# 2. 讀取資料
# ============================================================

print("\n========== Step 1: Loading Data ==========")

train = pd.read_excel(TRAIN_FILE)
valid = pd.read_excel(VALID_FILE)
test = pd.read_excel(TEST_FILE)
sample = pd.read_csv(SAMPLE_FILE)

print("Train shape:", train.shape)
print("Valid shape:", valid.shape)
print("Test shape:", test.shape)
print("Sample shape:", sample.shape)


# ============================================================
# 3. 資料前處理函式
# ============================================================

NUMERIC_COLS = [
    "土地移轉總面積平方公尺",
    "建物移轉總面積平方公尺",
    "車位移轉總面積平方公尺",
    "總樓層數",
    "建物現況格局-房",
    "建物現況格局-廳",
    "建物現況格局-衛"
]

FEATURE_COLS = [
    "縣市",
    "鄉鎮市區",
    "交易標的",
    "土地移轉總面積平方公尺",
    "都市土地使用分區",
    "交易年",
    "交易月",
    "總樓層數",
    "建物型態",
    "主要用途",
    "主要建材",
    "建物移轉總面積平方公尺",
    "建物現況格局-房",
    "建物現況格局-廳",
    "建物現況格局-衛",
    "建物現況格局-隔間",
    "有無管理組織",
    "車位類別",
    "車位移轉總面積平方公尺"
]


def to_number(series):
    """將文字或混合型態轉換成數值。"""
    return pd.to_numeric(
        series.astype("string").str.replace(
            ",", "", regex=False
        ).str.strip(),
        errors="coerce"
    ).astype("float64")


def preprocess(df):
    df = df.copy()

    # 3-1. 從完整地址提取縣市
    address = df["土地位置建物門牌"].fillna("").astype(str)

    df["縣市"] = address.str.extract(
        r"^(.{2,3}?[縣市])",
        expand=False
    ).fillna("未知")

    # 3-2. 處理民國交易日期
    date_str = (
        df["交易年月日"]
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(7)
    )

    year_roc = pd.to_numeric(
        date_str.str[:-4], errors="coerce"
    )

    month = pd.to_numeric(
        date_str.str[-4:-2], errors="coerce"
    )

    df["交易年"] = (year_roc + 1911).where(
        year_roc.between(80, 150)
    )

    df["交易月"] = month.where(
        month.between(1, 12)
    )

    # 3-3. 轉換數值欄位
    for col in NUMERIC_COLS:
        df[col] = to_number(df[col])

    # 3-4. 選取模型特徵
    df = df[FEATURE_COLS].copy()

    return df


# ============================================================
# 4. 建立 Train / Validation / Test
# ============================================================

print("\n========== Step 2: Preprocessing ==========")

train["總價元"] = to_number(train["總價元"])
valid["總價元"] = to_number(valid["總價元"])

# 移除沒有有效價格的資料
train = train.dropna(subset=["總價元"]).copy()
valid = valid.dropna(subset=["總價元"]).copy()

X_train = preprocess(train)
X_valid = preprocess(valid)
X_test = preprocess(test)

y_train = train["總價元"]
y_valid = valid["總價元"]

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("X_test:", X_test.shape)

print("\nFeature columns:")
print(X_train.columns.tolist())

print("\nMissing values:")
print(X_train.isnull().sum().to_string())


# ============================================================
# 5. 建立 Preprocessing Pipeline
# ============================================================

print("\n========== Step 3: Building Pipeline ==========")

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X_train.columns
    if col not in numeric_features
]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 數值資料：使用中位數補缺失值
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# 類別資料：填補缺失值 + One-Hot Encoding
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="未知"
    )),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=100
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ============================================================
# 6. 建立 Random Forest Regression Model
# ============================================================

print("\n========== Step 4: Training Model ==========")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Model training completed!")
print(f"Training time: {training_time:.2f} seconds")


# ============================================================
# 7. Validation Evaluation
# ============================================================

print("\n========== Step 5: Validation Evaluation ==========")

valid_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, valid_pred)
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
r2 = r2_score(y_valid, valid_pred)

print(f"MAE : {mae:,.2f} 元")
print(f"RMSE: {rmse:,.2f} 元")
print(f"R2  : {r2:.4f}")

# 儲存模型評估結果
metrics = pd.DataFrame([{
    "Model": "Random Forest",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Training_Time": training_time
}])

metrics.to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

# 儲存 Validation 的實際價格與預測價格
validation_results = pd.DataFrame({
    "Actual": y_valid.to_numpy(),
    "Predicted": valid_pred,
    "Absolute_Error": np.abs(
        y_valid.to_numpy() - valid_pred
    )
})

validation_results.to_csv(
    OUTPUT_DIR / "validation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nValidation examples:")
print(validation_results.head(10).to_string(index=False))


# ============================================================
# 8. 繪製 Actual vs Predicted
# ============================================================

print("\n========== Step 6: Visualization ==========")

plt.figure(figsize=(8, 6))

plt.scatter(
    y_valid,
    valid_pred,
    alpha=0.25,
    s=10
)

max_price = max(y_valid.max(), valid_pred.max())

plt.plot(
    [0, max_price],
    [0, max_price],
    color="red",
    linestyle="--"
)

plt.xlabel("Actual Price (NTD)")
plt.ylabel("Predicted Price (NTD)")
plt.title("Random Forest: Actual vs Predicted")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()

# 誤差分布
plt.figure(figsize=(8, 6))

errors = y_valid.to_numpy() - valid_pred

plt.hist(errors, bins=80)

plt.xlabel("Prediction Error (NTD)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_distribution.png",
    dpi=300
)

plt.close()

print("Evaluation figures saved!")


# ============================================================
# 9. 儲存模型
# ============================================================

print("\n========== Step 7: Saving Model ==========")

joblib.dump(
    model,
    OUTPUT_DIR / "random_forest_model.joblib"
)

print("Model saved!")


# ============================================================
# 10. 預測 Test
# ============================================================

print("\n========== Step 8: Test Prediction ==========")

test_pred = model.predict(X_test)

# 房價不可為負數
test_pred = np.maximum(test_pred, 0)

# 使用 Kaggle 提供的 Sample Submission 格式
if len(sample) != len(test_pred):
    raise ValueError(
        "Sample Submission 與 Test 筆數不相同，請確認資料版本"
    )

if "編號" not in sample.columns or "總價元" not in sample.columns:
    raise ValueError(
        "Sample Submission 欄位名稱不符合預期"
    )

# 保留 Sample Submission 的編號與欄位順序
submission = sample.copy()
submission["總價元"] = test_pred

submission.to_csv(
    BASE_DIR / "submission.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSubmission preview:")
print(submission.head().to_string(index=False))

print("\nSubmission shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())


# ============================================================
# 11. 完成
# ============================================================

print("\n===================================")
print("House Price Prediction Completed!")
print("===================================")
print(f"Validation MAE: {mae:,.2f} 元")
print(f"Submission saved: {BASE_DIR / 'submission.csv'}")
print(f"Outputs saved: {OUTPUT_DIR}")
