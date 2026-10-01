from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Approval Dashboard",
    page_icon="📊",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    # Chỉ load model do bạn huấn luyện và đưa vào repository.
    return joblib.load(APP_DIR / "best_model.joblib")


try:
    bundle = load_model()
except Exception as exc:
    st.error(f"Không load được model: {exc}")
    st.stop()

model = bundle["model"]
FEATURES = bundle["features"]
NUM_COLS = bundle["numeric_columns"]
CAT_COLS = bundle["categorical_columns"]
ALLOWED = bundle["allowed_categories"]


def clean_features(data):
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]

    if data.columns.duplicated().any():
        raise ValueError("CSV có tên cột trùng.")

    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

    result = data[FEATURES].copy()

    for col in FEATURES:
        result[col] = result[col].map(
            lambda value: (
                value.strip() if isinstance(value, str) else value
            )
        )

    result = result.replace({"?": np.nan, "": np.nan})
    result = result.where(result.notna(), np.nan)

    for col in NUM_COLS:
        result[col] = pd.to_numeric(
            result[col], errors="raise"
        ).astype(float)

        if not np.isfinite(result[col].dropna()).all():
            raise ValueError(f"{col} có giá trị không hữu hạn.")

    for col in CAT_COLS:
        result[col] = result[col].map(
            lambda value: (
                str(value) if pd.notna(value) else np.nan
            )
        )

        invalid = set(result[col].dropna()) - set(ALLOWED[col])
        if invalid:
            raise ValueError(
                f"{col} có mã không hợp lệ: {sorted(invalid)}"
            )

    return result


st.title("📊 Credit Approval Dashboard")
st.write(
    "Dự đoán xác suất chấp nhận (+) / từ chối (-) "
    "cho một nhóm hồ sơ có cùng mã hóa với dữ liệu UCI."
)

st.caption(
    "Dữ liệu UCI đã ẩn ý nghĩa A1–A15. "
    "Kết quả dùng cho thực hành phân tích, "
    "không phải quyết định phê duyệt tín dụng thực tế."
)

with st.sidebar:
    st.subheader("Mô hình")
    st.write(bundle["model_name"])

    threshold = st.slider(
        "Ngưỡng phân loại chấp nhận (+)",
        min_value=0.05,
        max_value=0.95,
        value=float(bundle.get("threshold", 0.50)),
        step=0.05
    )

    st.caption(
        "Ngưỡng thay đổi nhãn phân loại; "
        "không thay đổi xác suất và tỷ lệ kỳ vọng."
    )

    metrics = bundle.get("metadata", {}).get(
        "best_model_test_metrics", {}
    )
    if metrics:
        st.write("Kết quả trên tập test:")
        st.dataframe(
            pd.DataFrame(
                metrics.items(),
                columns=["Chỉ số", "Giá trị"]
            ),
            hide_index=True
        )

# CSV mẫu chỉ chứa tiêu đề, để người dùng biết cấu trúc cần nhập.
st.download_button(
    "Tải mẫu tiêu đề CSV",
    data=(",".join(FEATURES) + "\n").encode("utf-8-sig"),
    file_name="credit_input_template.csv",
    mime="text/csv"
)

uploaded = st.file_uploader(
    "Upload CSV có các cột A1 đến A15",
    type=["csv"]
)

if uploaded is None:
    st.info(
        "Chọn CSV để bắt đầu. Dữ liệu thiếu có thể để trống hoặc ghi '?'."
    )
    st.stop()

try:
    raw_data = pd.read_csv(
        uploaded,
        encoding="utf-8-sig",
        na_values=["?", ""]
    )

    if raw_data.empty:
        raise ValueError("CSV không có dòng dữ liệu.")

    # Kiểm tra cấu trúc trước khi dự đoán.
    raw_data.columns = [
        str(col).strip() for col in raw_data.columns
    ]

    n = int(st.number_input(
        "Số dòng cần dự đoán, lấy từ đầu CSV",
        min_value=1,
        max_value=len(raw_data),
        value=len(raw_data),
        step=1
    ))

    data = clean_features(raw_data.iloc[:n])

except Exception as exc:
    st.error(f"Dữ liệu đầu vào chưa hợp lệ: {exc}")
    st.stop()

st.caption(
    f"Dùng {n}/{len(raw_data)} dòng. "
    f"Có {int(data.isna().sum().sum())} ô thiếu; "
    "model xử lý theo quy tắc đã học khi huấn luyện."
)

warnings = []

for col in CAT_COLS:
    known = set(bundle["observed_categories"][col])
    unseen = set(data[col].dropna()) - known

    if unseen:
        warnings.append(
            f"{col}: mã hợp lệ nhưng chưa thấy khi huấn luyện "
            f"{sorted(unseen)}."
        )

for col in NUM_COLS:
    lower, upper = bundle["numeric_ranges"][col]
    outside = (data[col] < lower) | (data[col] > upper)

    if outside.any():
        warnings.append(
            f"{col}: {int(outside.sum())} dòng nằm ngoài "
            f"khoảng huấn luyện [{lower}, {upper}]."
        )

if warnings:
    st.warning(
        "Dữ liệu có khác biệt với tập huấn luyện. "
        "Xác suất ở các dòng này có thể kém tin cậy hơn."
    )
    with st.expander("Xem chi tiết"):
        for message in warnings:
            st.write(message)

try:
    positive_index = list(model.classes_).index(1)
    p_accept = model.predict_proba(data)[:, positive_index]
except Exception as exc:
    st.error(f"Không dự đoán được: {exc}")
    st.stop()

p_reject = 1 - p_accept
accept_mask = p_accept >= threshold

result = data.reset_index(drop=True).copy()
result.insert(0, "Dong_du_lieu", np.arange(1, n + 1))
result["P_chap_nhan"] = p_accept
result["P_tu_choi"] = p_reject
result["Nhan_du_doan"] = np.where(accept_mask, "+", "-")
result["Ket_qua"] = np.where(
    accept_mask, "Chấp nhận (+)", "Từ chối (-)"
)

col1, col2, col3 = st.columns(3)

col1.metric("Số dòng dự đoán", n)
col2.metric(
    "Tỷ lệ chấp nhận (+) kỳ vọng",
    f"{p_accept.mean():.2%}"
)
col3.metric(
    "Tỷ lệ từ chối (-) kỳ vọng",
    f"{p_reject.mean():.2%}"
)

summary = pd.DataFrame({
    "Kết quả": ["Chấp nhận (+)", "Từ chối (-)"],
    "Số dòng theo ngưỡng": [
        int(accept_mask.sum()),
        int((~accept_mask).sum())
    ],
    "Tỷ lệ theo ngưỡng (%)": [
        accept_mask.mean() * 100,
        (~accept_mask).mean() * 100
    ],
    "Số dòng kỳ vọng": [
        p_accept.sum(),
        p_reject.sum()
    ],
    "Tỷ lệ kỳ vọng (%)": [
        p_accept.mean() * 100,
        p_reject.mean() * 100
    ]
})

st.subheader("Tổng hợp dự đoán")
st.dataframe(summary.round(2), hide_index=True)

st.caption(
    "Tỷ lệ kỳ vọng là trung bình xác suất của các dòng. "
    "Tỷ lệ theo ngưỡng là tỷ lệ đếm sau khi phân loại."
)

left, right = st.columns(2)

with left:
    st.write("So sánh hai cách tính tỷ lệ")
    chart_data = summary.set_index("Kết quả")[[
        "Tỷ lệ kỳ vọng (%)",
        "Tỷ lệ theo ngưỡng (%)"
    ]]
    st.bar_chart(chart_data)

with right:
    st.write("Phân phối xác suất chấp nhận (+)")
    counts, edges = np.histogram(
        p_accept,
        bins=np.linspace(0, 1, 11)
    )
    histogram = pd.DataFrame({
        "Khoảng xác suất": [
            f"{edges[i]:.0%}–{edges[i + 1]:.0%}"
            for i in range(len(counts))
        ],
        "Số dòng": counts
    }).set_index("Khoảng xác suất")
    st.bar_chart(histogram)

st.subheader("Kết quả từng dòng")
st.dataframe(
    result.style.format({
        "P_chap_nhan": "{:.2%}",
        "P_tu_choi": "{:.2%}"
    })
)

st.download_button(
    "Tải kết quả dự đoán CSV",
    data=result.to_csv(index=False).encode("utf-8-sig"),
    file_name="credit_predictions.csv",
    mime="text/csv"
)

st.caption(
    "Nguồn huấn luyện: UCI Credit Approval — "
    "https://doi.org/10.24432/C5FS30"
)
