# pip install streamlit pandas numpy matplotlib openpyxl
# streamlit run app.py
# (İsteğe bağlı) Dosya olarak CSV/XLSX yükleyin; TR sayısal biçimleri otomatik çevrilir.
import io
from typing import Dict, Optional

import altair as alt

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    plt = None
import numpy as np
import pandas as pd
import streamlit as st


def describe_safe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        return df.describe(include="all")


def to_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace('%', '', regex=False)
    s = s.str.replace(r'[€₺$£]', '', regex=True).str.replace(' ', '', regex=False)

    def parse_one(x: str) -> float:
        if x in ('', 'nan', 'None', None):
            return np.nan
        if ',' in x and '.' in x:
            x = x.replace('.', '').replace(',', '.')
        elif ',' in x:
            x = x.replace(',', '.')
        try:
            return float(x)
        except Exception:
            return np.nan

    return s.map(parse_one)


def load_data(file_bytes: bytes, file_type: str, encoding: str, sep: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if file_type == "csv":
        return pd.read_csv(buffer, encoding=encoding, sep=sep, low_memory=False)
    if file_type == "excel":
        return pd.read_excel(buffer, engine="openpyxl")
    raise ValueError("Desteklenmeyen dosya türü")


def coerce_turkish_numbers(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    object_cols = result.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        converted = to_number(result[col])
        success_mask = converted.notna()
        success_ratio = success_mask.mean()
        unique_count = converted[success_mask].nunique(dropna=True)
        if success_ratio >= 0.7 and unique_count >= 5:
            result[col] = converted.astype(float)
    return result


def safe_infer_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    candidate_cols = result.select_dtypes(include=["object", "string"]).columns
    for col in candidate_cols:
        series = result[col].dropna()
        if series.empty:
            continue
        sample = series.astype(str).str.strip()
        contains_digits = sample.str.contains(r"\d", regex=True)
        contains_separators = sample.str.contains(r"[-/.]", regex=True)
        only_digits = sample.str.fullmatch(r"\d+", na=False)
        mask = contains_digits & contains_separators & ~only_digits
        if not mask.any():
            continue
        candidates = sample[mask]
        converted = pd.to_datetime(candidates, dayfirst=True, errors="coerce")
        if converted.notna().mean() >= 0.6:
            full_converted = pd.to_datetime(result[col], dayfirst=True, errors="coerce")
            if full_converted.notna().any():
                result[col] = full_converted
    return result


def _normalize(text: str) -> str:
    mapping = str.maketrans({
        "ç": "c",
        "Ç": "c",
        "ğ": "g",
        "Ğ": "g",
        "ı": "i",
        "İ": "i",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ü": "u",
        "Ü": "u",
        "â": "a",
        "Â": "a",
        "î": "i",
        "Î": "i",
        "û": "u",
        "Û": "u",
    })
    return text.translate(mapping).lower().strip()


def auto_guess_columns(columns) -> Dict[str, Optional[str]]:
    roles = {
        "musteri_kolonu": ["musteri", "musteriad", "customer", "firma"],
        "satis_muhendisi_kolonu": ["satismuh", "satismuhendisi", "sales", "temsilci"],
        "tahmini_tutar_kolonu": ["tahmini", "forecast", "tutar", "beklenen"],
        "olasilik_kolonu": ["olasilik", "olas", "prob", "ihtimal"],
        "agirlikli_tutar_kolonu": ["agirlik", "agirlikli", "weighted", "tutar"]
    }
    normalized_cols = {_normalize(col): col for col in columns}
    guesses: Dict[str, Optional[str]] = {role: None for role in roles}
    for role, keywords in roles.items():
        best_match = None
        best_score = 0
        for norm_col, original_col in normalized_cols.items():
            for kw in keywords:
                if kw in norm_col:
                    score = len(kw)
                    if score > best_score:
                        best_score = score
                        best_match = original_col
        guesses[role] = best_match
    return guesses


def plot_weighted_by_customer(df: pd.DataFrame, customer_col: str, weighted_col: str):
    filtered = df[[customer_col, weighted_col]].dropna()
    if filtered.empty:
        st.info("Grafik için yeterli veri bulunamadı (Müşteri/Ağırlıklı Tutar).")
        return
    filtered = filtered[filtered[customer_col].astype(str).str.strip() != ""]
    if filtered.empty:
        st.info("Müşteri bilgisi boş olduğu için grafik oluşturulamadı.")
        return
    grouped = (
        filtered.groupby(customer_col, dropna=False)[weighted_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    if grouped.empty:
        st.info("Grafik için yeterli veri bulunamadı (Müşteri/Ağırlıklı Tutar).")
        return
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(grouped[customer_col], grouped[weighted_col], color="#1f77b4")
        ax.set_title("Müşteriye Göre Toplam Ağırlıklı Tutar (Top 10)")
        ax.set_xlabel("Müşteri")
        ax.set_ylabel("Toplam Ağırlıklı Tutar (€)")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        chart = (
            alt.Chart(grouped)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X(f"{customer_col}:N", sort=list(grouped[customer_col]), title="Müşteri"),
                y=alt.Y(f"{weighted_col}:Q", title="Toplam Ağırlıklı Tutar (€)"),
                tooltip=[customer_col, weighted_col],
            )
        )
        st.altair_chart(chart, use_container_width=True)


def plot_weighted_by_engineer(df: pd.DataFrame, engineer_col: str, weighted_col: str):
    filtered = df[[engineer_col, weighted_col]].dropna()
    if filtered.empty:
        st.info("Grafik için yeterli veri bulunamadı (Satış Mühendisi/Ağırlıklı Tutar).")
        return
    filtered = filtered[filtered[engineer_col].astype(str).str.strip() != ""]
    if filtered.empty:
        st.info("Satış mühendisi bilgisi boş olduğu için grafik oluşturulamadı.")
        return
    grouped = (
        filtered.groupby(engineer_col, dropna=False)[weighted_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    if grouped.empty:
        st.info("Grafik için yeterli veri bulunamadı (Satış Mühendisi/Ağırlıklı Tutar).")
        return
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(grouped[engineer_col], grouped[weighted_col], color="#ff7f0e")
        ax.set_title("Satış Mühendisine Göre Toplam Ağırlıklı Tutar")
        ax.set_xlabel("Satış Mühendisi")
        ax.set_ylabel("Toplam Ağırlıklı Tutar (€)")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        chart = (
            alt.Chart(grouped)
            .mark_bar(color="#ff7f0e")
            .encode(
                x=alt.X(f"{engineer_col}:N", sort=list(grouped[engineer_col]), title="Satış Mühendisi"),
                y=alt.Y(f"{weighted_col}:Q", title="Toplam Ağırlıklı Tutar (€)"),
                tooltip=[engineer_col, weighted_col],
            )
        )
        st.altair_chart(chart, use_container_width=True)


def plot_estimated_amount_hist(df: pd.DataFrame, amount_col: str):
    data = df[amount_col].dropna()
    if data.empty:
        st.info("Histogram için yeterli Tahmini Tutar verisi bulunamadı.")
        return
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=15, color="#2ca02c", edgecolor="white")
        ax.set_title("Tahmini Tutar Dağılımı")
        ax.set_xlabel("Tahmini Tutar (€)")
        ax.set_ylabel("Frekans")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        chart = (
            alt.Chart(pd.DataFrame({amount_col: data}))
            .mark_bar(color="#2ca02c")
            .encode(
                x=alt.X(f"{amount_col}:Q", bin=alt.Bin(maxbins=15), title="Tahmini Tutar (€)"),
                y=alt.Y("count():Q", title="Frekans"),
                tooltip=["count():Q"],
            )
        )
        chart = chart.properties(title="Tahmini Tutar Dağılımı")
        st.altair_chart(chart, use_container_width=True)


def plot_probability_vs_estimate(df: pd.DataFrame, prob_col: str, amount_col: str):
    filtered = df[[prob_col, amount_col]].dropna()
    if filtered.empty:
        st.info("Saçılım grafiği için yeterli veri bulunamadı (Olasılık/Tahmini Tutar).")
        return
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(filtered[prob_col], filtered[amount_col], alpha=0.7, color="#d62728")
        ax.set_title("Olasılık (%) vs Tahmini Tutar (€)")
        ax.set_xlabel("Olasılık (%)")
        ax.set_ylabel("Tahmini Tutar (€)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        chart = (
            alt.Chart(filtered)
            .mark_circle(size=70, opacity=0.7, color="#d62728")
            .encode(
                x=alt.X(f"{prob_col}:Q", title="Olasılık (%)"),
                y=alt.Y(f"{amount_col}:Q", title="Tahmini Tutar (€)"),
                tooltip=[prob_col, amount_col],
            )
        )
        chart = chart.properties(title="Olasılık (%) vs Tahmini Tutar (€)")
        st.altair_chart(chart, use_container_width=True)


def determine_file_type(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith((".csv", ".txt")):
        return "csv"
    if name.endswith((".xls", ".xlsx")):
        return "excel"
    if "csv" in uploaded_file.type:
        return "csv"
    if "excel" in uploaded_file.type or "spreadsheet" in uploaded_file.type:
        return "excel"
    return None


def main():
    st.set_page_config(page_title="Satış Boru Hattı Analizi", layout="wide")
    st.title("Satış Boru Hattı Analizi")
    st.write(
        "Yüklediğiniz CSV veya XLSX dosyasındaki veriler TR sayı biçimleri temizlenerek analiz edilir."
    )

    st.header("Dosya Yükleme")
    st.write("CSV dosyaları için ayraç ve karakter kodlamasını aşağıdan seçebilirsiniz.")
    encoding = st.selectbox(
        "Encoding",
        options=["utf-8", "iso-8859-9", "windows-1254", "utf-16"],
        index=0,
        help="Dosya karakter kodlaması",
    )
    separator = st.selectbox(
        "Ayraç",
        options=[",", ";", "\t", "|"],
        index=0,
        help="CSV dosyalarındaki sütun ayraç karakteri",
    )
    uploaded_file = st.file_uploader("Dosya yükleyin", type=["csv", "xlsx", "xls", "txt"])

    if uploaded_file is None:
        st.info("Lütfen analiz etmek için bir CSV veya XLSX dosyası yükleyin.")
        return

    file_type = determine_file_type(uploaded_file)
    if file_type is None:
        st.error("Dosya türü tanımlanamadı. Lütfen CSV veya XLSX dosyası yükleyin.")
        return

    try:
        df = load_data(uploaded_file.getvalue(), file_type, encoding, separator)
    except Exception as exc:
        st.error(
            f"Dosya okunurken bir hata oluştu: {exc}. Lütfen ayraç veya encoding seçimini kontrol edin."
        )
        return

    if df.empty:
        st.warning("Yüklenen dosya boş görünüyor. Lütfen içeriğini kontrol edin.")
        return

    st.success("Dosya başarıyla yüklendi.")
    original_columns = df.columns.tolist()
    df = coerce_turkish_numbers(df)
    df = safe_infer_datetimes(df)

    st.header("Veri Özeti")
    st.write(f"Toplam satır: **{df.shape[0]}**, toplam sütun: **{df.shape[1]}**")
    preview = df.head(100)
    st.subheader("İlk 100 Satır Önizleme")
    st.dataframe(preview)

    st.subheader("Profil Özeti")
    try:
        profile = describe_safe(df).transpose().reset_index()
        profile.rename(columns={"index": "Özellik"}, inplace=True)
        st.dataframe(profile)
    except Exception as exc:
        st.error(f"Profil çıkarılırken bir hata oluştu: {exc}")

    st.header("Sütun Eşleştirme")
    guesses = auto_guess_columns(original_columns)
    column_mapping: Dict[str, Optional[str]] = {}
    for role, label in [
        ("musteri_kolonu", "Müşteri Sütunu"),
        ("satis_muhendisi_kolonu", "Satış Mühendisi Sütunu"),
        ("tahmini_tutar_kolonu", "Tahmini Tutar Sütunu (€)"),
        ("olasilik_kolonu", "Olasılık Sütunu (%)"),
        ("agirlikli_tutar_kolonu", "Ağırlıklı Tutar Sütunu (€)"),
    ]:
        options = ["(Seçiniz)"] + original_columns
        default = "(Seçiniz)"
        if guesses.get(role) in original_columns:
            default = guesses[role]
        selection = st.selectbox(label, options=options, index=options.index(default))
        column_mapping[role] = None if selection == "(Seçiniz)" else selection

    required_for_weighted = ["musteri_kolonu", "satis_muhendisi_kolonu", "agirlikli_tutar_kolonu"]
    missing_required = [role for role in required_for_weighted if column_mapping[role] is None]
    if missing_required:
        st.warning(
            "Ağırlıklı Tutar grafiklerini görebilmek için müşteri, satış mühendisi ve ağırlıklı tutar sütunlarını seçin."
        )

    if column_mapping["tahmini_tutar_kolonu"] is None:
        st.warning("Tahmini tutar sütunu seçilmezse dağılım grafiği gösterilemez.")
    if column_mapping["olasilik_kolonu"] is None:
        st.warning("Olasılık sütunu seçilmezse saçılım grafiği gösterilemez.")

    st.header("Grafikler")
    if column_mapping["musteri_kolonu"] and column_mapping["agirlikli_tutar_kolonu"]:
        plot_weighted_by_customer(
            df,
            column_mapping["musteri_kolonu"],
            column_mapping["agirlikli_tutar_kolonu"],
        )

    if column_mapping["satis_muhendisi_kolonu"] and column_mapping["agirlikli_tutar_kolonu"]:
        plot_weighted_by_engineer(
            df,
            column_mapping["satis_muhendisi_kolonu"],
            column_mapping["agirlikli_tutar_kolonu"],
        )

    if column_mapping["tahmini_tutar_kolonu"]:
        plot_estimated_amount_hist(df, column_mapping["tahmini_tutar_kolonu"])

    if column_mapping["olasilik_kolonu"] and column_mapping["tahmini_tutar_kolonu"]:
        plot_probability_vs_estimate(
            df,
            column_mapping["olasilik_kolonu"],
            column_mapping["tahmini_tutar_kolonu"],
        )


if __name__ == "__main__":
    main()
