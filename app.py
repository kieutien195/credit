from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import io
import tempfile

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


st.set_page_config(
    page_title="Credit Approval Dashboard",
    page_icon="📊",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
DRIVE_FILE_ID = "1GDFzbFXjbZMGdNmvy__wDmadeZId6ZfW"


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "best_model.joblib")


@st.cache_data(ttl=30, show_spinner=False)
def read_drive_csv(file_id):
    """Tải CSV công khai từ Drive; không dùng cookie đăng nhập."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "source.csv"

        downloaded = gdown.download(
            id=file_id,
            output=str(path),
            quiet=True,
            use_cookies=False
        )

        if downloaded is None:
            raise RuntimeError("Google Drive không cho tải file.")

        content = path.read_bytes()

    prefix = content.lstrip()[:500].lower()
    if b"<!doctype html" in prefix or b"<html" in prefix:
        raise ValueError("Drive trả về trang HTML thay vì CSV.")

    # Kiểm tra cột trùng trước khi pandas tự đổi tên.
    text = content.decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)), [])
    header = [name.strip() for name in header]

    if not header:
        raise ValueError("CSV không có tiêu đề.")

    if len(header) != len(set(header)):
        raise ValueError("CSV có tên cột bị trùng.")

    frame = pd.read_csv(
        io.StringIO(text),
        na_values=["?", ""]
    )

    if frame.empty:
        raise ValueError("CSV không có dòng dữ liệu.")

    checked_at = datetime.now(
        ZoneInfo("Asia/Ho_Chi_Minh")
    ).strftime("%d/%m/%Y %H:%M:%S")

    return frame, checked_at


try:
    bundle = load_model()
    model = bundle["model"]
    FEATURES = bundle["features"]
    NUM_COLS = bundle["numeric_columns"]
    CAT_COLS = bundle["categorical_columns"]
    ALLOWED = bundle["allowed_categories"]

except Exception as exc:
    st.error(f"Không load được best_model.joblib: {exc}")
    st.stop()


def clean_features(data):
    data = data.copy()
    data.columns = [str(col).strip() for col in data.columns]

    if data.columns.duplicated().any():
        raise ValueError("CSV có tên cột bị trùng.")

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
    "từ dữ liệu tự động đọc trên Google Drive."
)

st.caption(
    "Nguồn hiện tại: credit_new_10_synthetic.csv — "
    "dữ liệu giả lập để thử dự đoán, không phải hồ sơ thực tế."
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
        "Thay đổi ngưỡng chỉ thay đổi nhãn phân loại, "
        "không thay đổi xác suất hoặc tỷ lệ kỳ vọng."
    )

    st.subheader("Kết quả trên tập test")
    metrics = bundle.get("metadata", {}).get(
        "best_model_test_metrics", {}
    )

    if metrics:
        st.dataframe(
            pd.DataFrame(
                metrics.items(),
                columns=["Chỉ số", "Giá trị"]
            ).round(4),
            hide_index=True
        )

    st.caption(
        "Chấp nhận (+) = lớp 1\n\n"
        "Từ chối (-) = lớp 0"
    )

# Bộ đếm trên trình duyệt yêu cầu chạy lại app mỗi phút.
st_autorefresh(
    interval=60_000,
    debounce=False,
    key="credit_drive_refresh"
)

source_mode = st.radio(
    "Nguồn dữ liệu",
    options=[
        "File mẫu trên Google Drive",
        "Upload CSV của bạn"
    ],
    index=0,
    horizontal=True
)

try:
    if source_mode == "File mẫu trên Google Drive":
        st.info(
            "Đang dùng file mẫu trên Drive. "
            "Nguồn được đọc lại khoảng mỗi 60 giây khi trang đang mở."
        )

        if st.button("Đọc lại nguồn ngay"):
            read_drive_csv.clear()

        with st.spinner("Đang đọc file mẫu từ Google Drive..."):
            raw_data, checked_at = read_drive_csv(DRIVE_FILE_ID)

        st.caption(
            "File mẫu là dữ liệu giả lập để thử dự đoán. "
            f"Lần đọc nguồn thành công: {checked_at}"
        )

    else:
        st.info(
            "Upload CSV có tiêu đề A1 đến A15. "
            "Giá trị thiếu có thể để trống hoặc ghi '?'."
        )

        uploaded = st.file_uploader(
            "Chọn file CSV",
            type=["csv"],
            key="custom_credit_csv"
        )

        if uploaded is None:
            st.stop()

        # Kiểm tra cột trùng trước khi pandas tự đổi tên.
        text = uploaded.getvalue().decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)), [])
        header = [name.strip() for name in header]

        if not header:
            raise ValueError("CSV không có tiêu đề.")

        if len(header) != len(set(header)):
            raise ValueError("CSV có tên cột bị trùng.")

        raw_data = pd.read_csv(
            io.StringIO(text),
            na_values=["?", ""]
        )

        st.caption(f"File đã chọn: {uploaded.name}")

    if raw_data.empty:
        raise ValueError("CSV không có dòng dữ liệu.")

    use_all = st.checkbox(
        "Dự đoán tất cả các dòng",
        value=True,
        key=f"use_all_{source_mode}"
    )

    if use_all:
        n = len(raw_data)
    else:
        n = int(st.number_input(
            "Số dòng cần dự đoán, lấy từ đầu CSV",
            min_value=1,
            max_value=len(raw_data),
            value=len(raw_data),
            step=1,
            key=f"n_rows_{source_mode}_{len(raw_data)}"
        ))

    data = clean_features(raw_data.iloc[:n])

    with st.expander("Xem dữ liệu đầu vào"):
        st.dataframe(data, hide_index=True)

except Exception as exc:
    st.error(f"Không đọc được dữ liệu: {exc}")

    if source_mode == "File mẫu trên Google Drive":
        st.caption(
            "Kiểm tra file Drive đã bật "
            "Anyone with the link → Viewer. "
            "Bạn cũng có thể chuyển sang Upload CSV của bạn."
        )

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
            f"{col}: mã hợp lệ nhưng chưa thấy khi huấn luyện: "
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
        "Một số dữ liệu khác phạm vi đã quan sát khi huấn luyện; "
        "xác suất có thể kém tin cậy hơn."
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
    accept_mask,
    "Chấp nhận (+)",
    "Từ chối (-)"
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
    "https://doi.org/10.24432/C5FS30. "
    "Dữ liệu mới cần cùng mã hóa A1–A15. "
    "Dashboard cập nhật dự đoán, không tự huấn luyện lại model."
)
