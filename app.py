import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math

st.set_page_config(
    page_title="Data Wrangler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0f0f17 !important;
    color: #e2e2f0 !important;
}

.main .block-container {
    background: #0f0f17 !important;
    padding-top: 2rem !important;
}

section[data-testid="stSidebar"] {
    background: #13131f !important;
    border-right: 1px solid #2a2a3d !important;
}
section[data-testid="stSidebar"] * { color: #e2e2f0 !important; }

/* Expanders */
details {
    background: #16161f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 14px !important;
    margin-bottom: 12px !important;
    overflow: hidden !important;
}
details:hover { border-color: #4f46e5 !important; }
details[open] {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 1px #4f46e520, 0 8px 32px #4f46e510 !important;
}
details summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #c8c8e8 !important;
    padding: 16px 20px !important;
    cursor: pointer !important;
    letter-spacing: 0.5px !important;
    list-style: none !important;
}
details summary:hover { color: #a78bfa !important; }
details summary::-webkit-details-marker { display: none !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px #4f46e530 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px #4f46e550 !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #1e1e2e !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 8px !important;
    color: #e2e2f0 !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px #4f46e520 !important;
}

/* Selectbox & Multiselect */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #1e1e2e !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 8px !important;
    color: #e2e2f0 !important;
}

/* Slider */
.stSlider > div > div > div > div { background: #4f46e5 !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #2a2a3d !important; border-radius: 10px !important; overflow: hidden !important; }
thead tr th {
    background-color: #1e1e2e !important;
    color: #a78bfa !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #16161f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label { color: #6060a0 !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #a78bfa !important; font-family: 'Space Mono', monospace !important; font-size: 28px !important; }

/* Alerts */
div[data-testid="stAlert"] { border-radius: 8px !important; }

/* Radio & Checkbox labels */
.stRadio label, .stCheckbox label { color: #c8c8e8 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #16161f !important;
    border: 2px dashed #2a2a3d !important;
    border-radius: 14px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #4f46e5 !important; }

/* Download button */
.stDownloadButton > button {
    background: #16161f !important;
    color: #a78bfa !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #2a2a3d !important; }

/* General text */
p, label, span, div { color: #c8c8e8; }
h1, h2, h3, h4 { font-family: 'Space Mono', monospace !important; color: #e2e2f0 !important; }
</style>
""", unsafe_allow_html=True)

# ================= INIT STATE =================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "just_reset" not in st.session_state:
    st.session_state.just_reset = False
if "log" not in st.session_state:
    st.session_state.log = []

if "df" in st.session_state:
    df = st.session_state.df

# ================= SIDEBAR =================
st.sidebar.markdown("""
<div style='padding:24px 0 8px 0'>
    <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:3px;text-transform:uppercase;margin:0 0 16px 0'>Navigate</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox("", ["Upload & Overview", "Cleaning & Preparation"], label_visibility="collapsed")

st.sidebar.markdown("<hr style='border-color:#2a2a3d;margin:16px 0'>", unsafe_allow_html=True)

if "df" in st.session_state:
    _df = st.session_state.df
    st.sidebar.markdown(f"""
    <p style='font-family:Space Mono,monospace;font-size:10px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 12px 0'>Dataset</p>
    <div style='display:flex;flex-direction:column;gap:8px'>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #4f46e5'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Rows</span><br>
            <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.shape[0]:,}</span>
        </div>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #7c3aed'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Columns</span><br>
            <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.shape[1]}</span>
        </div>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #ef4444'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Missing</span><br>
            <span style='color:#f87171;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.isnull().sum().sum():,}</span>
        </div>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #f59e0b'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Duplicates</span><br>
            <span style='color:#fcd34d;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.duplicated().sum():,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style='margin-bottom:32px'>
    <h1 style='font-family:Space Mono,monospace;font-size:34px;font-weight:700;color:#e2e2f0;margin:0;line-height:1.1'>
        Data Wrangler <span style='color:#a78bfa'>&</span> Visualizer
    </h1>
    <p style='color:#7a7ac9;font-size:17px;margin:8px 0 0 0'>Clean, transform, and visualize your data effortlessly</p>
</div>
""", unsafe_allow_html=True)

# ================= PAGE A =================
if page == "Upload & Overview":

    cols_feat = st.columns(6)
    features = [
        ("📂", "Upload", "CSV, Excel, JSON"),
        ("🧹", "Clean", "Missing & duplicates"),
        ("📊", "Visualize", "Interactive charts"),
        ("🔄", "Transform", "Scale & modify"),
        ("⚙️", "Process", "Advanced ops"),
        ("📤", "Export", "Download results"),
    ]
    for i, (icon, title, sub) in enumerate(features):
        with cols_feat[i]:
            st.markdown(f"""
            <div style='background:#16161f;border:1px solid #2a2a3d;border-radius:12px;padding:16px 10px;text-align:center;margin-bottom:24px'>
                <div style='font-size:22px;margin-bottom:6px'>{icon}</div>
                <div style='font-family:Space Mono,monospace;font-size:12px;font-weight:700;color:#c8c8e8'>{title}</div>
                <div style='font-size:12px;color:#8a8ad0;margin-top:4px'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Step 01</p>
    <h2 style='font-family:Space Mono,monospace;font-size:20px;color:#e2e2f0;margin:0 0 16px 0'>Upload Your Dataset</h2>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your file here — CSV, Excel, or JSON",
        type=["csv", "xlsx", "json"],
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)
        st.session_state.df = df.copy()

    if "df" not in st.session_state:
        if st.session_state.get("just_reset", False):
            st.success("Session reset. Please upload your dataset again.")
            st.session_state.just_reset = False
    else:
        df = st.session_state.df

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Step 02</p>
        <h2 style='font-family:Space Mono,monospace;font-size:20px;color:#e2e2f0;margin:0 0 16px 0'>Dataset Overview</h2>
        """, unsafe_allow_html=True)

        st.caption("Preview of your uploaded data (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True, height=300)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])

        def simplify_dtype(dtype):
            if "int" in str(dtype) or "float" in str(dtype):
                return "Numeric"
            elif "datetime" in str(dtype):
                return "Date"
            else:
                return "Text"

        col_types = pd.DataFrame({
            "Column": df.columns,
            "Type": [simplify_dtype(dtype) for dtype in df.dtypes]
        })
        st.dataframe(col_types, use_container_width=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:16px 0 8px 0'>Step 03</p>
        <h2 style='font-family:Space Mono,monospace;font-size:20px;color:#e2e2f0;margin:0 0 16px 0'>Summary Statistics</h2>
        """, unsafe_allow_html=True)

        with st.expander("📊 Numeric Summary", expanded=False):
            num_df = df.select_dtypes(include=np.number)
            if not num_df.empty:
                st.dataframe(num_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns available")

        with st.expander("🏷️ Categorical Summary", expanded=False):
            cat_df = df.select_dtypes(include=["object", "category"])
            if not cat_df.empty:
                summary_cat = pd.DataFrame({
                    "Column": cat_df.columns,
                    "Unique Values": [cat_df[col].nunique() for col in cat_df.columns],
                    "Most Frequent": [cat_df[col].mode()[0] if not cat_df[col].mode().empty else None for col in cat_df.columns]
                })
                st.dataframe(summary_cat, use_container_width=True)
            else:
                st.info("No categorical columns available")

        with st.expander("⚠️ Missing Values", expanded=False):
            missing = df.isnull().sum()
            percent = (missing / len(df)) * 100
            st.dataframe(pd.DataFrame({"Missing": missing, "%": percent}), use_container_width=True)

        with st.expander("👥 Duplicates", expanded=False):
            st.markdown(f"""
            <div style='background:#1e1e2e;border-radius:10px;padding:20px;text-align:center'>
                <p style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin:0 0 4px 0'>Duplicate Rows Found</p>
                <p style='color:#fcd34d;font-family:Space Mono,monospace;font-size:36px;font-weight:700;margin:0'>{df.duplicated().sum():,}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Session"):
            current_key = st.session_state.get("uploader_key", 0)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.uploader_key = current_key + 1
            st.rerun()

# ================= PAGE B =================
elif page == "Cleaning & Preparation":

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload & Overview page.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns

        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Page B</p>
        <h2 style='font-family:Space Mono,monospace;font-size:22px;color:#e2e2f0;margin:0 0 4px 0'>Cleaning Tools</h2>
        <p style='color:#6060a0;font-size:13px;margin:0 0 24px 0'>Click any section below to expand it and apply transformations.</p>
        """, unsafe_allow_html=True)

        # ===== TYPE CONVERSION =====
        with st.expander("🔄  Data Type Conversion", expanded=False):
            col = st.selectbox("Column", df.columns)
            dtype = st.selectbox("Convert to", ["Numeric", "Datetime", "Categorical"])
            fmt = st.text_input("Datetime format (optional)") if dtype == "Datetime" else None
            clean_option = []
            if dtype == "Numeric":
                clean_option = st.multiselect(
                    "Clean numeric issues",
                    ["Remove commas (,)", "Remove currency symbols ($, €, £)", "Remove spaces"]
                )
            if st.button("Apply Conversion"):
                if dtype == "Numeric":
                    temp = df[col].astype(str)
                    if "Remove commas (,)" in clean_option:
                        temp = temp.str.replace(",", "")
                    if "Remove currency symbols ($, €, £)" in clean_option:
                        temp = temp.str.replace(r"[€$£]", "", regex=True)
                    if "Remove spaces" in clean_option:
                        temp = temp.str.replace(" ", "")
                    df[col] = pd.to_numeric(temp, errors="coerce")
                elif dtype == "Datetime":
                    df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
                elif dtype == "Categorical":
                    df[col] = df[col].astype("category")
                st.session_state.df = df
                st.session_state.log.append(f"Converted {col} to {dtype}")
                st.success("Conversion applied")

        # ===== MISSING VALUES =====
        with st.expander("🩹  Missing Values Handling", expanded=False):
            missing = df.isnull().sum()
            percent = (missing / len(df)) * 100
            st.dataframe(pd.DataFrame({"Missing": missing, "%": percent}), use_container_width=True)

            col = st.selectbox("Select column to handle missing values", df.columns, key="missing_col")
            action = st.selectbox(
                "Action",
                ["Drop Rows", "Mean", "Median", "Mode", "Most Frequent", "Constant", "Forward Fill", "Backward Fill"],
                key="missing_action"
            )
            val = st.text_input("Constant value", key="missing_val") if action == "Constant" else None
            before_rows = len(df)
            before_missing = df[col].isnull().sum()

            if st.button("Apply Missing Handling"):
                if before_missing == 0:
                    st.info("No missing values in this column. Nothing to handle.")
                else:
                    if action in ["Mean", "Median"] and col not in num_cols:
                        st.error("Mean/Median can only be applied to numeric columns")
                    else:
                        if action == "Drop Rows":
                            df = df.dropna(subset=[col])
                        elif action == "Mean":
                            df[col] = df[col].fillna(df[col].mean())
                        elif action == "Median":
                            df[col] = df[col].fillna(df[col].median())
                        elif action in ["Mode", "Most Frequent"]:
                            df[col] = df[col].fillna(df[col].mode()[0])
                        elif action == "Constant":
                            df[col] = df[col].fillna(val)
                        elif action == "Forward Fill":
                            df[col] = df[col].ffill()
                        elif action == "Backward Fill":
                            df[col] = df[col].bfill()

                        after_rows = len(df)
                        after_missing = df[col].isnull().sum()
                        st.success("Missing values handled")
                        st.write("### Before vs After")
                        st.write(f"Rows: {before_rows} → {after_rows}")
                        st.write(f"Missing in '{col}': {before_missing} → {after_missing}")
                        st.write("### Updated Missing Values")
                        missing_new = df.isnull().sum()
                        percent_new = (missing_new / len(df)) * 100
                        st.dataframe(pd.DataFrame({"Missing": missing_new, "%": percent_new}), use_container_width=True)
                        st.session_state.df = df
                        st.session_state.log.append(f"Handled missing in {col} using {action}")

        # ===== DROP COLUMNS BY MISSING % =====
        with st.expander("🗑️  Drop Columns by Missing %", expanded=False):
            st.write("### Current Missing %")
            before_missing_pct = (df.isnull().mean() * 100)
            st.dataframe(before_missing_pct.to_frame(name="% Missing"), use_container_width=True)
            threshold = st.slider("Drop columns above % missing", 0, 100, 50)
            if st.button("Drop Columns"):
                before_cols = df.shape[1]
                to_drop = df.columns[df.isnull().mean() * 100 > threshold]
                if len(to_drop) == 0:
                    st.info("No columns exceeded the threshold. Nothing was dropped.")
                else:
                    df = df.drop(columns=to_drop)
                    after_cols = df.shape[1]
                    st.success("Columns dropped")
                    st.write("### Summary")
                    st.write(f"Columns: {before_cols} → {after_cols}")
                    st.write("Dropped columns:", list(to_drop))
                    st.write("### Remaining Columns")
                    after_missing_pct = (df.isnull().mean() * 100)
                    st.dataframe(after_missing_pct.to_frame(name="% Missing"), use_container_width=True)
                    st.session_state.df = df
                    st.session_state.log.append(f"Dropped columns > {threshold}% missing")

        # ===== DROP SELECTED COLUMNS =====
        with st.expander("✂️  Drop Selected Columns", expanded=False):
            cols_to_drop = st.multiselect("Select columns to drop", df.columns, key="drop_selected_cols")
            if st.button("Drop Selected Columns", key="drop_selected_btn"):
                if cols_to_drop:
                    before_cols = df.shape[1]
                    df = df.drop(columns=cols_to_drop)
                    after_cols = df.shape[1]
                    st.session_state.df = df
                    st.session_state.log.append(f"Dropped columns: {cols_to_drop}")
                    st.success(f"Dropped {before_cols - after_cols} column(s)")
                    st.write(f"Columns: {before_cols} → {after_cols}")
                    st.write("Dropped:", cols_to_drop)
                else:
                    st.warning("Select at least one column to drop")

        # ===== DUPLICATES =====
        with st.expander("👥  Duplicates Handling", expanded=False):
            mode = st.radio("Duplicate type", ["Full Row", "Subset"])
            if mode == "Subset":
                cols = st.multiselect("Select columns", df.columns)
                if not cols:
                    st.warning("Please select at least one column.")
            else:
                cols = None

            if mode == "Full Row":
                dup = df.duplicated()
                dup_all = df[df.duplicated(keep=False)]
            elif mode == "Subset" and cols:
                dup = df.duplicated(subset=cols)
                dup_all = df[df.duplicated(subset=cols, keep=False)]
            else:
                dup = pd.Series([False] * len(df))
                dup_all = pd.DataFrame()

            st.write(f"Duplicates found: {dup.sum()}")
            if st.checkbox("Show duplicate rows"):
                if not dup_all.empty:
                    dup_all = dup_all.copy()
                    if mode == "Full Row":
                        dup_all["Duplicate_Group"] = dup_all.groupby(list(df.columns)).ngroup()
                    else:
                        dup_all["Duplicate_Group"] = dup_all.groupby(cols).ngroup()
                    st.dataframe(dup_all.sort_values("Duplicate_Group"), use_container_width=True)
                else:
                    st.write("No duplicates found.")

            keep = st.selectbox("Keep", ["First", "Last"])
            if st.button("Remove Duplicates"):
                before = len(df)
                if mode == "Full Row":
                    df = df.drop_duplicates(keep="first" if keep == "First" else "last")
                elif mode == "Subset" and cols:
                    df = df.drop_duplicates(subset=cols, keep="first" if keep == "First" else "last")
                after = len(df)
                st.write(f"Rows: {before} → {after}")
                st.session_state.df = df
                st.session_state.log.append(f"Removed duplicates ({mode}, keep {keep})")
                st.success("Duplicates removed")

        # ===== CATEGORICAL TOOLS =====
        with st.expander("🏷️  Categorical Tools", expanded=False):
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) > 0:
                c = st.selectbox("Categorical column", cat_cols)
                std_action = st.selectbox("Standardize", ["None", "Lower", "Upper", "Title", "Trim"])
                if st.button("Apply Standardization"):
                    if std_action == "None":
                        st.warning("Please select a standardization option.")
                    else:
                        if std_action == "Lower":
                            df[c] = df[c].str.lower()
                        elif std_action == "Upper":
                            df[c] = df[c].str.upper()
                        elif std_action == "Title":
                            df[c] = df[c].str.title()
                        elif std_action == "Trim":
                            df[c] = df[c].str.strip()
                        st.session_state.df = df
                        st.session_state.log.append(f"Standardized {c}")
                        st.success("Standardization applied")
                        st.dataframe(df[[c]].head(10), use_container_width=True)

                mapping_text = st.text_area("Mapping old:new")
                if st.button("Apply Mapping"):
                    mapping_dict = {}
                    for line in mapping_text.split("\n"):
                        if ":" in line:
                            k, v = line.split(":")
                            mapping_dict[k.strip()] = v.strip()
                    if not mapping_dict:
                        st.warning("Please enter valid mappings in format old:new")
                    else:
                        df[c] = df[c].map(mapping_dict).fillna(df[c])
                        st.session_state.df = df
                        st.session_state.log.append(f"Mapped values in {c}")
                        st.success("Mapping applied")
                        st.dataframe(df[[c]].head(10), use_container_width=True)

                thresh = st.slider("Rare category threshold %", 0, 20, 5)
                freq = df[c].value_counts(normalize=True) * 100
                rare = freq[freq < thresh].index
                selected = st.multiselect("Select categories to group into 'Other'", options=list(rare))
                if st.button("Group Rare"):
                    if len(selected) == 0:
                        st.info("No categories selected")
                    else:
                        before_counts = df[c].value_counts()
                        df[c] = df[c].replace(selected, "Other")
                        after_counts = df[c].value_counts()
                        st.session_state.df = df
                        st.session_state.log.append(f"Grouped rare in {c} (<{thresh}%)")
                        st.success("Rare categories grouped")
                        st.write("### Before")
                        st.dataframe(before_counts.head(10), use_container_width=True)
                        st.write("### After")
                        st.dataframe(after_counts.head(10), use_container_width=True)

                if st.button("One-hot Encode"):
                    df = pd.get_dummies(df, columns=[c])
                    st.session_state.df = df
                    st.session_state.log.append(f"Encoded {c}")
                    st.success("One-hot encoding applied")
                    st.dataframe(df.head(10), use_container_width=True)
            else:
                st.info("No categorical columns available")

        # ===== OUTLIERS =====
        with st.expander("📈  Outlier Detection & Handling", expanded=False):
            if len(num_cols) > 0:
                c = st.selectbox("Select numeric column", num_cols, key="outlier_col")
                Q1 = df[c].quantile(0.25)
                Q3 = df[c].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[c] < lower) | (df[c] > upper)]
                st.markdown(f"""
                <div style='background:#1e1e2e;border-radius:10px;padding:16px;margin-bottom:12px;display:flex;gap:24px'>
                    <div><span style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Outliers</span><br>
                    <span style='color:#f87171;font-family:Space Mono,monospace;font-size:22px;font-weight:700'>{len(outliers):,}</span></div>
                    <div><span style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Lower Bound</span><br>
                    <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:22px;font-weight:700'>{lower:.2f}</span></div>
                    <div><span style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Upper Bound</span><br>
                    <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:22px;font-weight:700'>{upper:.2f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                action = st.selectbox("Choose action", ["Do Nothing", "Remove Rows", "Cap (Winsorize)"], key="outlier_action")
                if st.button("Apply Outlier Handling"):
                    before_rows = len(df)
                    if action == "Do Nothing":
                        st.info("No changes applied")
                    elif len(outliers) == 0:
                        st.info("No outliers detected — no changes needed")
                    else:
                        if action == "Remove Rows":
                            df = df[(df[c] >= lower) & (df[c] <= upper)]
                        elif action == "Cap (Winsorize)":
                            df[c] = np.where(df[c] < lower, lower, np.where(df[c] > upper, upper, df[c]))
                        after_rows = len(df)
                        st.session_state.df = df
                        st.session_state.log.append(f"Outlier handling on {c}: {action}")
                        st.success("Outlier handling applied")
                        st.write("### Impact")
                        st.write(f"Outliers detected: {len(outliers)}")
                        st.write(f"Rows: {before_rows} → {after_rows}")
            else:
                st.info("No numeric columns available")

        # ===== SCALING =====
        with st.expander("⚖️  Scaling", expanded=False):
            if len(num_cols) > 0:
                c = st.selectbox("Select numeric column to scale", num_cols, key="scale_col")
                m = st.selectbox("Method", ["MinMax", "Z-score"], key="scale_method")
                before_mean = df[c].mean()
                before_std = df[c].std()
                if st.button("Apply Scaling"):
                    scaler = MinMaxScaler() if m == "MinMax" else StandardScaler()
                    df[[c]] = scaler.fit_transform(df[[c]])
                    after_mean = df[c].mean()
                    after_std = df[c].std()
                    st.session_state.df = df
                    st.session_state.log.append(f"Scaled {c} using {m}")
                    st.success("Scaling applied")
                    st.write("### Before vs After")
                    st.write(f"Mean: {before_mean:.2f} → {after_mean:.2f}")
                    st.write(f"Std Dev: {before_std:.2f} → {after_std:.2f}")
            else:
                st.info("No numeric columns available")

        # ===== COLUMN OPERATIONS =====
        with st.expander("🔧  Column Operations", expanded=False):
            st.write("### Rename Column")
            old_name = st.selectbox("Select column to rename", df.columns, key="rename_col")
            new_name = st.text_input("New column name", key="rename_input")
            if st.button("Rename Column"):
                if new_name:
                    df = df.rename(columns={old_name: new_name})
                    st.session_state.df = df
                    st.session_state.log.append(f"Renamed {old_name} to {new_name}")
                    st.success("Column renamed")
                else:
                    st.warning("Enter a new name")

            st.markdown("<hr style='border-color:#2a2a3d;margin:20px 0'>", unsafe_allow_html=True)
            st.write("### Create New Column")
            col1 = st.selectbox("Column A", df.columns, key="colA")
            col2 = st.selectbox("Column B", df.columns, key="colB")
            operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide", "Log(A)", "A - Mean(A)"], key="operation")
            new_col = st.text_input("New column name", key="create_col_input")
            if st.button("Create Column"):
                if new_col:
                    if operation == "Add":
                        df[new_col] = df[col1] + df[col2]
                    elif operation == "Subtract":
                        df[new_col] = df[col1] - df[col2]
                    elif operation == "Multiply":
                        df[new_col] = df[col1] * df[col2]
                    elif operation == "Divide":
                        df[new_col] = df[col1] / df[col2]
                    elif operation == "Log(A)":
                        df[new_col] = np.log(df[col1].replace(0, np.nan))
                    elif operation == "A - Mean(A)":
                        df[new_col] = df[col1] - df[col1].mean()
                    st.session_state.df = df
                    st.session_state.log.append(f"Created {new_col}")
                    st.success("New column created")
                    st.dataframe(df[[new_col]].head(), use_container_width=True)
                else:
                    st.warning("Enter column name")

            st.markdown("<hr style='border-color:#2a2a3d;margin:20px 0'>", unsafe_allow_html=True)
            st.write("### Binning")
            if len(num_cols) > 0:
                col_bin = st.selectbox("Select numeric column for binning", num_cols, key="bin_col")
                bins = st.slider("Number of bins", 2, 10, 4, key="bin_slider")
                method = st.selectbox("Method", ["Equal Width", "Quantile"], key="bin_method")
                bin_col_name = st.text_input("New binned column name", key="bin_input")
                if st.button("Apply Binning"):
                    if not bin_col_name:
                        st.warning("Enter column name")
                    else:
                        try:
                            if method == "Equal Width":
                                binned = pd.cut(df[col_bin], bins=bins)
                            else:
                                binned = pd.qcut(df[col_bin], q=bins, duplicates="drop")
                            labels = []
                            for interval in binned.cat.categories:
                                left = round(interval.left, 2)
                                right = round(interval.right, 2)
                                labels.append(f"{left} - {right}")
                            if method == "Equal Width":
                                df[bin_col_name] = pd.cut(df[col_bin], bins=bins, labels=labels)
                            else:
                                df[bin_col_name] = pd.qcut(df[col_bin], q=bins, labels=labels, duplicates="drop")
                            st.session_state.df = df
                            st.session_state.log.append(f"Binned {col_bin} into {bins} bins ({method})")
                            st.success("Binning applied")
                            st.write("### Preview")
                            st.dataframe(df[[col_bin, bin_col_name]].head(10), use_container_width=True)
                            st.write("### Bin Counts")
                            st.dataframe(df[bin_col_name].value_counts(), use_container_width=True)
                        except Exception as e:
                            st.error(f"Binning failed: {e}")
            else:
                st.info("No numeric columns available")

        # ===== DATA VALIDATION RULES =====
        with st.expander("✅  Data Validation Rules", expanded=False):
            validation_type = st.selectbox(
                "Select validation type",
                ["Numeric Range", "Allowed Categories", "Non-null Constraint"],
                key="val_type"
            )
            col_val = st.selectbox("Select column", df.columns, key="val_col")
            violations = pd.DataFrame()

            if validation_type == "Numeric Range":
                min_val = st.number_input("Minimum value", value=float(df[col_val].min()), key="val_min")
                max_val = st.number_input("Maximum value", value=float(df[col_val].max()), key="val_max")
            elif validation_type == "Allowed Categories":
                allowed = st.text_input("Enter allowed values (comma-separated)", key="val_allowed")

            check = st.button("Check Violations", key="val_button")
            if check:
                if validation_type == "Numeric Range":
                    if col_val not in num_cols:
                        st.error("Selected column is not numeric")
                    elif min_val > max_val:
                        st.error("Minimum cannot be greater than maximum")
                        st.stop()
                    else:
                        violations = df[(df[col_val] < min_val) | (df[col_val] > max_val)]
                elif validation_type == "Allowed Categories":
                    allowed_list = [x.strip() for x in allowed.split(",")]
                    violations = df[~df[col_val].isin(allowed_list)]
                elif validation_type == "Non-null Constraint":
                    violations = df[df[col_val].isnull()]

                if not violations.empty:
                    st.error(f"Violations found: {len(violations)}")
                    st.dataframe(violations, use_container_width=True)
                    csv = violations.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Violations CSV",
                        data=csv,
                        file_name="violations.csv",
                        mime="text/csv",
                        key="val_download"
                    )
                else:
                    st.success("No violations found")
