# pip install streamlit pandas numpy plotly openpyxl pyarrow scipy requests
# streamlit run app.py
# export OPENAI_API_KEY=...
import io
import os
import sys
import json
import textwrap
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


CUSTOM_CSS = """
<style>
    .main {background-color: #f9fafc;}
    .stApp {padding-top: 1rem;}
    h1, h2, h3 {color: #253858;}
    .section-divider {border-bottom: 1px solid #d9dde3; margin: 1.5rem 0;}
    .download-links a {margin-right: 1rem; color: #1f77b4; font-weight: 600;}
</style>
"""


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_type: str, encoding: str = "utf-8", sep: str = ",") -> pd.DataFrame:
    if file_type == "csv":
        data_io = io.BytesIO(file_bytes)
        return pd.read_csv(data_io, encoding=encoding, sep=sep, low_memory=False)
    if file_type == "excel":
        data_io = io.BytesIO(file_bytes)
        return pd.read_excel(data_io, engine="openpyxl")
    raise ValueError("Desteklenmeyen dosya türü.")


def detect_file_type(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith((".csv", ".txt")):
        return "csv"
    if name.endswith((".xls", ".xlsx")):
        return "excel"
    if "csv" in uploaded_file.type:
        return "csv"
    if "excel" in uploaded_file.type or "spreadsheet" in uploaded_file.type:
        return "excel"
    return ""


def infer_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    working_df = df.copy()
    numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = []
    for col in working_df.columns:
        if pd.api.types.is_datetime64_any_dtype(working_df[col]):
            datetime_cols.append(col)
        elif working_df[col].dtype == object:
            sample = working_df[col].dropna().head(200)
            if not sample.empty:
                parsed = pd.to_datetime(sample, errors="coerce", utc=True)
                if parsed.notna().mean() > 0.8:
                    datetime_cols.append(col)
                    working_df[col] = pd.to_datetime(working_df[col], errors="coerce")
    categorical_cols = [col for col in working_df.columns if col not in numeric_cols + datetime_cols]
    return {
        "numeric": numeric_cols,
        "datetime": datetime_cols,
        "categorical": categorical_cols,
    }


def profile_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = df.isna().mean().reset_index()
    missing.columns = ["Sütun", "Eksik Oranı"]
    basic_stats = df.describe(include="all", datetime_is_numeric=True).transpose().reset_index()
    basic_stats.rename(columns={"index": "Sütun"}, inplace=True)
    return missing, basic_stats


def summarize_dataframe(df: pd.DataFrame, schema: Dict[str, List[str]]) -> str:
    buffer = ["Veri Özeti:"]
    buffer.append(f"Toplam satır: {df.shape[0]}, toplam sütun: {df.shape[1]}")
    buffer.append("Sütun tipleri:")
    for key, cols in schema.items():
        buffer.append(f"  - {key}: {', '.join(cols) if cols else 'Yok'}")
    missing, _ = profile_data(df)
    top_missing = missing.sort_values("Eksik Oranı", ascending=False).head(10)
    buffer.append("Eksik değer oranları (ilk 10):")
    for _, row in top_missing.iterrows():
        buffer.append(f"  - {row['Sütun']}: {row['Eksik Oranı']:.2%}")
    return "\n".join(buffer)


def suggest_charts(df: pd.DataFrame, schema: Dict[str, List[str]]) -> Tuple[str, Dict[str, str], str]:
    reason = ""
    config: Dict[str, str] = {}
    if schema["datetime"] and schema["numeric"]:
        reason = "Zaman serisi ve sayısal sütunlar tespit edildiği için çizgi grafik önerildi."
        config = {"x": schema["datetime"][0], "y": schema["numeric"][0]}
        return "line", config, reason
    if len(schema["numeric"]) == 1:
        reason = "Tek sayısal sütun bulundu; dağılımı incelemek için histogram önerildi."
        config = {"x": schema["numeric"][0]}
        return "hist", config, reason
    if len(schema["numeric"]) >= 3:
        corr = df[schema["numeric"]].corr().abs()
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_corr = upper_tri.max().max()
        if max_corr is not None and max_corr > 0.5:
            reason = "Birden fazla sayısal sütun ve güçlü korelasyon bulundu; korelasyon ısı haritası önerildi."
            return "heatmap", {}, reason
    if len(schema["numeric"]) >= 2:
        reason = "İki sayısal sütun seçilerek ilişkilerini göstermek için saçılım grafiği önerildi."
        config = {"x": schema["numeric"][0], "y": schema["numeric"][1]}
        return "scatter", config, reason
    if schema["categorical"] and schema["numeric"]:
        cat_col = sorted(schema["categorical"], key=lambda c: df[c].nunique() if c in df else 0)[0]
        reason = "Kategorik ve sayısal sütun kombinasyonu için çubuk grafik önerildi."
        config = {"x": cat_col, "y": schema["numeric"][0]}
        return "bar", config, reason
    if len(schema["categorical"]) >= 2:
        reason = "İki kategorik sütun bulundu; yığılmış çubuk grafik önerildi."
        config = {"x": schema["categorical"][0], "color": schema["categorical"][1]}
        return "bar", config, reason
    return "table", {}, "Uygun bir grafik tipi belirlenemedi; tablo gösterilecektir."


def prepare_plot(df: pd.DataFrame, chart_type: str, config: Dict[str, str], top_n: int) -> Optional[go.Figure]:
    if chart_type == "line" and config.get("x") and config.get("y"):
        sorted_df = df.sort_values(config["x"]).dropna(subset=[config["x"], config["y"]])
        if sorted_df.empty:
            return None
        return px.line(sorted_df, x=config["x"], y=config["y"], markers=True)
    if chart_type == "scatter" and config.get("x") and config.get("y"):
        scatter_df = df.dropna(subset=[config["x"], config["y"]])
        if scatter_df.empty:
            return None
        return px.scatter(scatter_df, x=config["x"], y=config["y"], color=config.get("color"))
    if chart_type == "bar" and config.get("x") and config.get("y"):
        bar_df = df.dropna(subset=[config["x"], config["y"]])
        if bar_df.empty:
            return None
        if config.get("color") is None:
            nunique = bar_df[config["x"]].nunique()
            if nunique > top_n:
                agg = bar_df.groupby(config["x"])[config["y"]].mean().nlargest(top_n)
                bar_df = agg.reset_index()
        else:
            nunique = bar_df[config["x"]].nunique()
            if nunique > top_n:
                top_values = bar_df.groupby(config["x"])[config["y"]].mean().nlargest(top_n).index
                bar_df = bar_df[bar_df[config["x"]].isin(top_values)]
        return px.bar(bar_df, x=config["x"], y=config["y"], color=config.get("color"))
    if chart_type == "hist" and config.get("x"):
        hist_df = df.dropna(subset=[config["x"]])
        if hist_df.empty:
            return None
        return px.histogram(hist_df, x=config["x"], marginal="box")
    if chart_type == "box":
        target = config.get("y") or config.get("x")
        if target:
            box_df = df.dropna(subset=[target])
            if box_df.empty:
                return None
            return px.box(box_df, y=target)
    if chart_type == "heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return None
        corr = df[numeric_cols].corr()
        if corr.empty:
            return None
        return px.imshow(corr, text_auto=True, color_continuous_scale="Blues")
    return None


def generate_sample_rows(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    if df.shape[0] <= limit:
        return df
    return df.sample(limit, random_state=42)


def build_prompt(file_name: str, df: pd.DataFrame, schema: Dict[str, List[str]], chart_reason: str) -> str:
    missing, stats = profile_data(df)
    sample = generate_sample_rows(df, limit=20)
    sample_json = sample.to_json(orient="records", force_ascii=False)
    prompt = f"""
    Sen bir veri analizi asistanısın. Kullanıcı bir veri seti yükledi ve özet bilgi istiyor.

    Veri seti adı: {file_name}
    Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}

    Sütun tipleri:
    {json.dumps(schema, ensure_ascii=False, indent=2)}

    Eksik değer oranları:
    {missing.to_json(orient="records", force_ascii=False)}

    Temel istatistikler (ilk 10 sütun):
    {stats.head(10).to_json(orient="records", force_ascii=False)}

    Örnek satırlar (en fazla 20):
    {sample_json}

    Grafik seçimi gerekçesi: {chart_reason}

    Eğer kişisel veri olabileceğini düşünüyorsan bunu belirt ve maskeleme öner.

    Görev: 300-500 kelimelik Türkçe bir rapor hazırla. İçerikte:
    - Öne çıkan bulgular ve dağılımlar
    - Olası anomaliler veya dikkat çeken noktalar
    - İşe dönük 3-5 öneri
    - Ek analiz önerileri
    """
    prompt = textwrap.dedent(prompt).strip()
    return prompt


def run_llm_analysis(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Uzman bir veri analisti olarak konuş."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1200,
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        st.error(f"LLM analizi başarısız oldu: {exc}")
        return ""


def download_button(label: str, content: bytes, file_name: str, mime: str):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{file_name}">{label}</a>'
    st.markdown(f"<div class='download-links'>{href}</div>", unsafe_allow_html=True)


def apply_filters(df: pd.DataFrame, schema: Dict[str, List[str]]) -> pd.DataFrame:
    filtered_df = df.copy()
    st.subheader("Filtreler")
    with st.expander("Filtreleri Göster", expanded=False):
        filter_col = st.selectbox("Sütun seç", options=["(Yok)"] + df.columns.tolist(), index=0)
        if filter_col and filter_col != "(Yok)":
            if filter_col in schema["numeric"]:
                min_val = float(filtered_df[filter_col].min()) if filtered_df[filter_col].notna().any() else 0.0
                max_val = float(filtered_df[filter_col].max()) if filtered_df[filter_col].notna().any() else 0.0
                selected = st.slider("Aralık", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(filtered_df[filter_col] >= selected[0]) & (filtered_df[filter_col] <= selected[1])]
            elif filter_col in schema["datetime"]:
                min_date = filtered_df[filter_col].min()
                max_date = filtered_df[filter_col].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    selected = st.date_input("Tarih aralığı", value=(min_date.date(), max_date.date()))
                    if isinstance(selected, tuple) and len(selected) == 2:
                        start, end = selected
                        filtered_df = filtered_df[
                            (filtered_df[filter_col] >= pd.Timestamp(start))
                            & (filtered_df[filter_col] <= pd.Timestamp(end))
                        ]
                else:
                    st.info("Bu tarih sütununda geçerli değer bulunamadı.")
            else:
                unique_vals = filtered_df[filter_col].dropna().unique().tolist()
                selected_vals = st.multiselect(
                    "Değer seç", options=unique_vals, default=unique_vals[: min(len(unique_vals), 10)]
                )
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[filter_col].isin(selected_vals)]
    return filtered_df


def main() -> None:
    st.set_page_config(page_title="Veri Analiz Asistanı", layout="wide")

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Başlık ve açıklamalar
    st.title("Veri Analiz Asistanı")
    st.markdown("Kendi verinizi yükleyerek otomatik özet, grafik önerileri ve LLM tabanlı analiz raporu alın.")

    with st.expander("Nasıl Kullanılır?", expanded=True):
        st.markdown(
            """
            1. CSV veya Excel formatında dosyanızı yükleyin (50-100 MB'a kadar destek).
            2. Kodlama ve ayraç seçeneklerini gerektiğinde değiştirin.
            3. Önerilen grafiği inceleyin veya farklı grafik türü seçin.
            4. Filtreler ve dönüşümlerle veri alt kümelerini analiz edin.
            5. LLM analiziyle kısa bir rapor üretin ve indirin.
            """
        )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.header("Dosya Yükleme")
    col_upload, col_options = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Dosya yükleyin", type=["csv", "txt", "xls", "xlsx"], accept_multiple_files=False)
    with col_options:
        encoding = st.text_input("Kodlama", value="utf-8")
        sep = st.text_input("Ayraç (CSV için)", value=",")

    file_type = detect_file_type(uploaded_file)

    if uploaded_file:
        try:
            file_bytes = uploaded_file.getvalue()
            data_frame = load_data(file_bytes, file_type, encoding=encoding, sep=sep)
            st.success(f"Dosya başarıyla yüklendi: {uploaded_file.name}")
        except UnicodeDecodeError:
            st.error("Kodlama hatası. Lütfen doğru kodlamayı seçin (örn. UTF-8, ISO-8859-9).")
            data_frame = None
        except ValueError as exc:
            st.error(str(exc))
            data_frame = None
        except Exception as exc:
            st.error(f"Dosya okunamadı: {exc}")
            data_frame = None
    else:
        data_frame = None

    if data_frame is not None:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.header("Veri Özeti")
        schema = infer_schema(data_frame)
        summary_text = summarize_dataframe(data_frame, schema)
        st.text(summary_text)

        missing, stats = profile_data(data_frame)
        st.subheader("Eksik Değer Yüzdeleri")
        st.dataframe(missing.style.format({"Eksik Oranı": "{:.2%}"}))

        st.subheader("Temel İstatistikler")
        st.dataframe(stats.head(50))

        st.subheader("Örnek Satırlar (ilk 100)")
        st.dataframe(data_frame.head(100))

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.header("Grafik Önerici & Çizici")

        filtered_df = apply_filters(data_frame, schema)

        suggestion, config, reason = suggest_charts(filtered_df, schema)
        st.info(f"Önerilen grafik: {suggestion}. Gerekçe: {reason}")

        chart_type_options = ["line", "bar", "scatter", "hist", "box", "heatmap", "table"]
        default_chart_index = chart_type_options.index(suggestion) if suggestion in chart_type_options else 0
        chart_type = st.selectbox("Grafik Türü", options=chart_type_options, index=default_chart_index)

        columns_list = filtered_df.columns.tolist()
        options_with_none = ["(Yok)"] + columns_list

        default_x = config.get("x") if config.get("x") in columns_list else "(Yok)"
        default_y = config.get("y") if config.get("y") in columns_list else "(Yok)"
        default_color = config.get("color") if config.get("color") in columns_list else "(Yok)"

        col_x, col_y, col_color = st.columns(3)
        with col_x:
            x_axis = st.selectbox("X ekseni", options=options_with_none, index=options_with_none.index(default_x))
        with col_y:
            y_axis = st.selectbox("Y ekseni", options=options_with_none, index=options_with_none.index(default_y))
        with col_color:
            color_axis = st.selectbox("Renk/Kategori", options=options_with_none, index=options_with_none.index(default_color))

        top_n = st.slider("Top N (kategorik özet)", min_value=5, max_value=50, value=10, step=1)

        manual_config: Dict[str, str] = {}
        if x_axis != "(Yok)":
            manual_config["x"] = x_axis
        if y_axis != "(Yok)":
            manual_config["y"] = y_axis
        if color_axis != "(Yok)":
            manual_config["color"] = color_axis

        final_config = {**config, **manual_config}

        if chart_type == "table":
            st.dataframe(filtered_df.head(100))
        else:
            fig = prepare_plot(filtered_df, chart_type, final_config, top_n)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                try:
                    png_bytes = fig.to_image(format="png")
                except Exception:
                    png_bytes = None
                html_bytes = fig.to_html(full_html=False).encode("utf-8")
                st.subheader("Grafik İndirme")
                if png_bytes:
                    download_button("PNG olarak indir", png_bytes, "grafik.png", "image/png")
                download_button("HTML olarak indir", html_bytes, "grafik.html", "text/html")
            else:
                st.warning("Grafik oluşturulamadı. Lütfen eksen seçimlerini kontrol edin veya uygun sütunları seçin.")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.header("LLM Analiz Raporu")
        api_key_available = bool(os.getenv("OPENAI_API_KEY"))
        if not api_key_available:
            st.warning("OPENAI_API_KEY tanımlı değil. Lütfen ortam değişkeni ekleyin ya da bu bölümü atlayın.")

        prompt = build_prompt(uploaded_file.name, filtered_df, schema, reason)
        report_placeholder = st.empty()
        if st.button("Analizi Çalıştır", disabled=not api_key_available):
            with st.spinner("LLM analizi hazırlanıyor..."):
                report = run_llm_analysis(prompt)
                if report:
                    report_placeholder.markdown(report)
                    st.subheader("Rapor İndir")
                    download_button("Markdown indir", report.encode("utf-8"), "analiz_raporu.md", "text/markdown")
                else:
                    report_placeholder.warning("Rapor üretilemedi.")
    else:
        st.info("Analize başlamak için lütfen bir dosya yükleyin.")


def _is_streamlit_context_active() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit_context_active():
    main()
elif __name__ == "__main__":
    try:
        from streamlit.web import bootstrap

        script_path = Path(__file__).resolve()
        command_line = f"{script_path} {' '.join(sys.argv[1:])}".strip()
        bootstrap.run(str(script_path), command_line, sys.argv[1:])
    except Exception as exc:
        message = (
            "Streamlit uygulaması doğrudan Python ile başlatılamadı. "
            "Lütfen `streamlit run app.py` komutunu kullanın.\n"
            f"Ayrıntılar: {exc}"
        )
        sys.stderr.write(message + "\n")
