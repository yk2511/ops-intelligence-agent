import streamlit as st
import pandas as pd
import os
import io
import plotly.graph_objects as go
import plotly.express as px
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_anthropic import ChatAnthropic
from datetime import datetime, date

# ── Supabase (via requests — no C++ required) ───────────────────────────────
import requests
import json

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
SUPABASE_OK  = bool(SUPABASE_URL and SUPABASE_KEY)

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal"
}

def _clean_table(name: str) -> str:
    t = name.lower().replace(".csv","").replace(".xlsx","").replace(".xls","")
    t = ''.join(c if c.isalnum() or c == '_' else '_' for c in t)
    return t.strip('_')

def table_exists(table: str) -> bool:
    """Check if a table exists in Supabase."""
    try:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}?limit=1",
                         headers=SB_HEADERS, timeout=15)
        return r.status_code == 200
    except Exception:
        return False

def get_create_sql(table: str, df: pd.DataFrame) -> str:
    """Generate CREATE TABLE SQL for any dataframe — all columns as text."""
    col_defs = ["id bigserial primary key"]
    for col in df.columns:
        safe = col.replace('"', '').replace("'", "")
        col_defs.append('"' + safe + '" text')
    col_defs += ['"_upload_date" text', '"_uploaded_at" text']
    return 'create table if not exists "' + table + '" (' + ', '.join(col_defs) + ');'

def upload_to_supabase(df: pd.DataFrame, table_name: str, upload_date: str) -> dict:
    """Upload a dataframe to Supabase. If table missing, returns SQL to create it."""
    try:
        table    = _clean_table(table_name)
        r_test   = requests.get(f"{SUPABASE_URL}/rest/v1/{table}?limit=1",
                                 headers=SB_HEADERS, timeout=15)

        if r_test.status_code not in (200, 201):
            sql = get_create_sql(table, df)
            return {"success": False, "new_table": True,
                    "table": table, "create_sql": sql, "message": "new_table"}

        df_up = df.copy()
        df_up["_upload_date"] = upload_date
        df_up["_uploaded_at"] = datetime.utcnow().isoformat()

        for col in df_up.columns:
            df_up[col] = df_up[col].fillna("").astype(str)
            try:
                mask = df_up[col].str.match(r"^\d+\.0$")
                if mask.any():
                    df_up[col] = df_up[col].str.replace(r"\.0$", "", regex=True)
            except Exception:
                pass

        records  = df_up.to_dict(orient="records")
        url      = f"{SUPABASE_URL}/rest/v1/{table}"
        total    = 0

        for i in range(0, len(records), 50):
            batch = records[i:i+50]
            for attempt in range(3):
                try:
                    r = requests.post(url, headers=SB_HEADERS,
                                      data=json.dumps(batch), timeout=90)
                    if r.status_code in (200, 201):
                        total += len(batch)
                        break
                    else:
                        if attempt == 2:
                            return {"success": False,
                                    "message": f"HTTP {r.status_code}: {r.text[:200]}"}
                except requests.exceptions.ConnectionError:
                    if attempt == 2:
                        raise
                    import time; time.sleep(2)

        return {"success": True, "rows": total, "table": table}
    except Exception as e:
        return {"success": False, "message": str(e)}

def fetch_from_supabase(table_name: str) -> pd.DataFrame:
    """Fetch all rows from a Supabase table using REST API with retry."""
    try:
        table    = _clean_table(table_name)
        url      = f"{SUPABASE_URL}/rest/v1/{table}?select=*"
        all_rows = []
        offset   = 0
        limit    = 500  # smaller page size to avoid timeout

        while True:
            for attempt in range(3):
                try:
                    r = requests.get(
                        f"{url}&limit={limit}&offset={offset}",
                        headers=SB_HEADERS, timeout=60
                    )
                    if r.status_code == 200:
                        break
                except requests.exceptions.ConnectionError:
                    import time; time.sleep(2)
            else:
                break  # all 3 attempts failed — stop fetching

            if r.status_code != 200:
                break

            batch = r.json()
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < limit:
                break
            offset += limit

        if all_rows:
            df   = pd.DataFrame(all_rows)
            meta = [c for c in df.columns if c.startswith("_") or c == "id"]
            return df.drop(columns=meta, errors="ignore")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_upload_history(table_name: str) -> list:
    """Get distinct upload dates for a table."""
    try:
        table = _clean_table(table_name)
        url   = f"{SUPABASE_URL}/rest/v1/{table}?select=_upload_date"
        r     = requests.get(url, headers=SB_HEADERS, timeout=15)
        if r.status_code == 200 and r.json():
            dates = list(set(row.get("_upload_date","") for row in r.json()))
            return sorted([d for d in dates if d], reverse=True)
        return []
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Ops Intelligence Agent", layout="wide", page_icon="🏭")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Professional Styling
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Font & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide Streamlit default chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Top Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,179,237,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.header-banner h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #93c5fd;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 500;
}
.header-tag {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    color: #93c5fd;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.75rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1e3a8a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── KPI Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    border-color: #3b82f6;
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #1e293b;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    padding: 0.85rem 1.25rem !important;
    font-size: 0.9rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #60a5fa !important;
    background: rgba(59,130,246,0.05) !important;
    border-radius: 12px;
}

/* ── Selectboxes ── */
[data-testid="stSelectbox"] label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    transition: all 0.2s !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #cbd5e1 !important;
}
[data-testid="stButton"] button:hover {
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
    background: rgba(59,130,246,0.1) !important;
    transform: translateY(-1px);
}
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
    border: 1px solid #1e293b !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
}

/* ── Dividers ── */
hr {
    border-color: #1e293b !important;
    margin: 1.5rem 0 !important;
}

/* ── Info / Warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-size: 0.85rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #334155 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border-radius: 10px !important;
    border: 2px dashed #334155 !important;
    padding: 0.5rem !important;
}

/* ── Suggestion pill buttons ── */
.suggestion-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <h1>🏭 Raw Material Optimization Dashboard</h1>
    <p>AI-powered operations intelligence for plastic injection moulding plants</p>
    <span class="header-tag">⚡ Powered by Claude AI &nbsp;|&nbsp; MBA Pilot Project</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD + CLOUD SYNC
# ══════════════════════════════════════════════════════════════════════════════
has_files = 'user_files' in st.session_state and bool(st.session_state.user_files)

with st.expander("📂 File Manager & Cloud Sync", expanded=not has_files):
    # ── Two tabs: Upload Today's Data | Load Historical Data ─────────────────
    tab_upload, tab_history = st.tabs(["📤 Upload & Save to Cloud", "☁️ Load Historical Data"])

    with tab_upload:
        st.markdown("Upload today's files. They will be **saved to Supabase cloud** and appended to historical data.")

        up_col1, up_col2 = st.columns([3, 2])
        with up_col1:
            uploaded_files = st.file_uploader(
                "Upload production data",
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
                key="user_files",
                label_visibility="collapsed"
            )
        with up_col2:
            upload_date = st.date_input("Data Date", value=date.today(),
                                        help="Select the date this data represents")
            save_to_cloud = st.button("☁️ Save to Cloud", type="primary",
                                      use_container_width=True,
                                      disabled=not SUPABASE_OK)
            if not SUPABASE_OK:
                st.caption("Run: `pip install supabase` to enable cloud sync")

    with tab_history:
        st.markdown("Load all historical data stored in Supabase cloud for trend analysis.")
        load_history = st.button("📥 Load All Historical Data", type="primary",
                                  use_container_width=True, disabled=not SUPABASE_OK)
        if SUPABASE_OK:
            st.caption("This loads ALL data ever uploaded — across all days — for full trend analysis.")
        else:
            st.caption("Run: `pip install supabase` to enable cloud sync")

@st.cache_data
def load_file(file_bytes, file_name):
    if file_name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        return pd.read_excel(io.BytesIO(file_bytes))

# ── Use session_state to persist data across reruns ──────────────────────────
if "all_dfs" not in st.session_state:
    all_dfs = {}


# ── Initialize all_dfs ─────────────────────────────────────────────────────
all_dfs = {}

# ── Handle file uploads ──────────────────────────────────────────────────────
if uploaded_files:
    for f in uploaded_files:
        raw = f.read()
        all_dfs[f.name] = load_file(raw, f.name)

    # Save to Supabase if button clicked
    if save_to_cloud and SUPABASE_OK:
        with st.spinner("Saving to Supabase cloud..."):
            results = []
            for fname, df in all_dfs.items():
                table = fname.replace('.csv','').replace('.xlsx','').replace('.xls','')
                try:
                    clean_t = _clean_table(table)
                    del_url = f"{SUPABASE_URL}/rest/v1/{clean_t}?_upload_date=eq.{upload_date}"
                    requests.delete(del_url, headers=SB_HEADERS, timeout=30)
                except Exception:
                    pass
                result = upload_to_supabase(df, table, str(upload_date))
                results.append((fname, result))

            new_tables = [r for _, r in results if not r["success"] and r.get("new_table")]
            failed     = [r for _, r in results if not r["success"] and not r.get("new_table")]
            succeeded  = [(f, r) for f, r in results if r["success"]]

            if succeeded:
                total_rows = sum(r["rows"] for _, r in succeeded)
                st.success(f"✅ {len(succeeded)} file(s) saved — {total_rows:,} rows for {upload_date}")
                st.balloons()
            if new_tables:
                st.warning(f"⚠️ {len(new_tables)} new table(s) detected — run SQL below, then upload again.")
                for r in new_tables:
                    with st.expander(f"📋 SQL to create: {r['table']}"):
                        st.code(r["create_sql"], language="sql")
            if failed:
                for _, r in failed:
                    st.error(f"❌ {r['message']}")

    st.toast(f"✅ {len(st.session_state['all_dfs'])} file(s) loaded!", icon="🚀")
    st.caption(f"✓ **Active Files:** {', '.join(st.session_state['all_dfs'].keys())}")

# ── Load historical data from Supabase ──────────────────────────────────────
elif load_history and SUPABASE_OK:
    with st.spinner("Loading historical data from Supabase..."):
        known_tables = [
            "01_raw_material_master", "02_part_master", "03_production_orders",
            "04_mold_changeover_log", "05_customer_orders", "06_machine_shift_log",
            "07_material_consumption", "08_purchase_orders_grn", "09_raw_material_inventory"
        ]
        loaded_count = 0
        for table in known_tables:
            df_hist = fetch_from_supabase(table)
            if not df_hist.empty:
                all_dfs[f"{table}.csv"] = df_hist
                loaded_count += 1

        if loaded_count > 0:
            st.session_state["cloud_data"] = all_dfs
            total_rows = sum(len(v) for v in all_dfs.values())
            st.success(f"✅ Loaded {loaded_count} tables from cloud "
                       f"({total_rows:,} total rows across all history!)")
            with st.expander("📅 Upload History"):
                for table in known_tables:
                    if f"{table}.csv" in all_dfs:
                        dates = get_upload_history(table)
                        if dates:
                            st.caption(f"**{table}** — {len(dates)} upload(s): "
                                       f"{', '.join(dates[:5])}"
                                       f"{'...' if len(dates) > 5 else ''}")
        else:
            st.warning("No historical data found in cloud yet. Upload files first!")

# ── Restore from session state if already loaded ────────────────────────────
elif "cloud_data" in st.session_state and st.session_state["cloud_data"]:
    all_dfs = st.session_state["cloud_data"]
    st.caption(f"☁️ **Cloud data active:** {len(all_dfs)} tables — "
               f"{sum(len(v) for v in all_dfs.values()):,} rows")

else:
    # Fallback: local files
    local_files = ['bom_master.csv', 'production_logs.csv', 'scrap_logs.csv']
    for fname in local_files:
        try:
            all_dfs[fname] = pd.read_csv(fname)
        except FileNotFoundError:
            pass
    if all_dfs:
        st.caption(f"📁 **Using local files:** {', '.join(all_dfs.keys())}")
    else:
        st.warning("⬆️ Please upload at least one CSV file to get started.")
        st.stop()

if not all_dfs:
    st.warning("⬆️ Please upload files or load historical data to continue.")
    st.stop()

if not all_dfs:
    st.warning("⬆️ Please upload files or load historical data to continue.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PLANT HEALTH METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Plant Health Metrics</div>', unsafe_allow_html=True)

req_files_scorecard = [
    '07_material_consumption.csv',
    '01_raw_material_master.csv',
    '03_production_orders.csv',
    '09_raw_material_inventory.csv'
]

if all(f in all_dfs for f in req_files_scorecard):
    try:
        df_mc  = all_dfs['07_material_consumption.csv']
        df_rm  = all_dfs['01_raw_material_master.csv']
        df_po  = all_dfs['03_production_orders.csv']
        df_inv = all_dfs['09_raw_material_inventory.csv']

        avg_efficiency   = df_mc['Material_Utilization_Pct'].mean()

        df_mc_cost = pd.merge(df_mc, df_rm[['Material_Code', 'Unit_Cost_INR_per_kg']], on='Material_Code', how='left')
        df_mc_cost['Waste_Cost_INR'] = (df_mc_cost['Scrap_Qty_kg'] + df_mc_cost['Purge_Qty_kg']) * df_mc_cost['Unit_Cost_INR_per_kg']
        total_waste_cost = df_mc_cost['Waste_Cost_INR'].sum()

        total_wos = len(df_po['WO_ID'].unique())

        df_inv_alert  = pd.merge(df_inv, df_rm[['Material_Code', 'MOQ_kg']], on='Material_Code', how='left')
        critical_alerts = len(df_inv_alert[df_inv_alert['Qty_Available_kg'] < df_inv_alert['MOQ_kg']])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Material Efficiency",   f"{avg_efficiency:.1f}%")
        c2.metric("Logged Waste Cost",          f"₹ {total_waste_cost:,.0f}",
                  delta="Needs Review", delta_color="inverse")
        c3.metric("Processed Work Orders",      f"{total_wos}")
        c4.metric("Critical Inventory Alerts",  f"{critical_alerts} items",
                  delta="Check Silos", delta_color="inverse")

    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
else:
    st.info("💡 Upload files 07, 01, 03, and 09 to activate the Plant Health Metrics.")

# ══════════════════════════════════════════════════════════════════════════════
# RAW DATA VIEWER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("")
with st.expander("🔍 View Uploaded Data"):
    tabs = st.tabs(list(all_dfs.keys()))
    for i, (fname, frame) in enumerate(all_dfs.items()):
        with tabs[i]:
            st.caption(f"**{fname}** — {frame.shape[0]:,} rows × {frame.shape[1]} columns")
            st.dataframe(frame, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SANKEY DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
if all(f in all_dfs for f in ['07_material_consumption.csv', '03_production_orders.csv', '02_part_master.csv']):
    st.markdown("")
    with st.expander("🌊 View Raw Material Flow (Sankey Diagram)", expanded=False):
        st.markdown("Trace material from the **Silo → Finished Part** to identify invisible losses.")
        try:
            df_mc  = all_dfs['07_material_consumption.csv']
            df_po  = all_dfs['03_production_orders.csv']
            df_bom = all_dfs['02_part_master.csv']

            df_po_bom = pd.merge(df_po, df_bom[['Part_Code', 'Material_per_Part_g']], on='Part_Code', how='left')
            df_po_bom['Theoretical_Good_kg'] = (df_po_bom['Actual_Good_Parts'] * df_po_bom['Material_per_Part_g']) / 1000
            df_flow = pd.merge(df_mc, df_po_bom[['WO_ID', 'Theoretical_Good_kg']], on='WO_ID', how='left')

            total_issued          = df_flow['Qty_Issued_kg'].sum()
            total_returned        = df_flow['Returned_to_Store_kg'].sum()
            total_processed       = total_issued - total_returned
            total_scrap           = df_flow['Scrap_Qty_kg'].sum()
            total_purge           = df_flow['Purge_Qty_kg'].sum()
            total_good_theoretical = df_flow['Theoretical_Good_kg'].sum()
            total_unaccounted     = total_processed - (total_good_theoretical + total_scrap + total_purge)

            labels = [
                "Material Issued (Silo)", "Returned to Store", "Net Material Processed",
                "Theoretical Good Parts", "Logged Scrap (Visible)", "Purging Waste",
                "Unaccounted Variance (Ghost Leak)"
            ]
            node_colors = ["#1d4ed8","#475569","#2563eb","#10b981","#f59e0b","#f97316","#ef4444"]

            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=25, thickness=30,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    label=labels,
                    color=node_colors
                ),
                link=dict(
                    source=[0, 0, 2, 2, 2, 2],
                    target=[1, 2, 3, 4, 5, 6],
                    value=[total_returned, total_processed, total_good_theoretical,
                           total_scrap, total_purge, total_unaccounted],
                    color=["rgba(71,85,105,0.4)", "rgba(37,99,235,0.3)", "rgba(16,185,129,0.3)",
                           "rgba(245,158,11,0.4)", "rgba(249,115,22,0.4)", "rgba(239,68,68,0.4)"]
                )
            )])

            fig.update_layout(
                title_text="Plant-Wide Mass Balance (Hover for KG values)",
                title_font=dict(size=14, color="#e2e8f0"),
                font=dict(size=12, color="#cbd5e1"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
                margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary row below Sankey
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Issued", f"{total_issued:,.0f} kg")
            s2.metric("Good Parts (Theoretical)", f"{total_good_theoretical:,.0f} kg")
            s3.metric("Logged Scrap + Purge", f"{(total_scrap + total_purge):,.0f} kg")
            s4.metric("Ghost Leak (Unaccounted)", f"{total_unaccounted:,.0f} kg",
                      delta="Investigate" if total_unaccounted > 0 else "✅ Clean",
                      delta_color="inverse" if total_unaccounted > 0 else "normal")

        except Exception as e:
            st.warning(f"Could not generate Sankey Diagram. Ensure files 07, 03, and 02 are uploaded. Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SCRAP PARETO
# ══════════════════════════════════════════════════════════════════════════════
if '06_machine_shift_log.csv' in all_dfs:
    st.markdown("")
    with st.expander("📊 View Scrap Pareto Analysis (80/20 Rule)", expanded=False):
        st.markdown("Identifies the **vital few defects** causing the most scrap — focus here first.")
        try:
            df_shift  = all_dfs['06_machine_shift_log.csv']
            defect_df = df_shift.groupby('Rejection_Reason')['Rejected_Parts'].sum().reset_index()
            defect_df = defect_df.sort_values(by='Rejected_Parts', ascending=False)
            defect_df['Cumulative_Pct'] = (defect_df['Rejected_Parts'].cumsum() / defect_df['Rejected_Parts'].sum()) * 100

            # Find 80% cutoff
            cutoff_idx = defect_df[defect_df['Cumulative_Pct'] <= 80].index.tolist()

            bar_colors = ['#f59e0b' if i in cutoff_idx else '#475569' for i in defect_df.index]

            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(
                x=defect_df['Rejection_Reason'],
                y=defect_df['Rejected_Parts'],
                name='Rejected Parts',
                marker_color=bar_colors,
                hovertemplate='<b>%{x}</b><br>Rejected: %{y:,.0f}<extra></extra>'
            ))
            fig_pareto.add_trace(go.Scatter(
                x=defect_df['Rejection_Reason'],
                y=defect_df['Cumulative_Pct'],
                name='Cumulative %',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='#ef4444', width=2.5),
                marker=dict(size=7, color='#ef4444'),
                hovertemplate='Cumulative: %{y:.1f}%<extra></extra>'
            ))
            # 80% reference line
            fig_pareto.add_hline(
                y=80, yref='y2',
                line_dash="dash", line_color="rgba(239,68,68,0.4)", line_width=1.5,
                annotation_text="80% threshold",
                annotation_font=dict(color="#ef4444", size=11)
            )
            fig_pareto.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(color="#cbd5e1"),
                margin=dict(l=20, r=20, t=30, b=60),
                yaxis=dict(
                    title='Rejected Parts (Count)',
                    gridcolor='rgba(51,65,85,0.5)',
                    title_font=dict(color="#94a3b8")
                ),
                yaxis2=dict(
                    title='Cumulative %',
                    overlaying='y', side='right',
                    range=[0, 105],
                    gridcolor='rgba(0,0,0,0)',
                    title_font=dict(color="#94a3b8")
                ),
                xaxis=dict(tickangle=-30, gridcolor='rgba(0,0,0,0)'),
                showlegend=False,
                hovermode="x unified"
            )
            st.plotly_chart(fig_pareto, use_container_width=True)
            st.caption("🟡 **Amber bars** = defects causing 80% of rejections — prioritise these for corrective action.")

        except Exception as e:
            st.warning(f"Could not generate Pareto. Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# WEEK VS WEEK TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("")
with st.expander("📊 Week vs Week Trend Analysis", expanded=False):
    st.markdown("Compare key metrics across weeks to identify improving or worsening trends.")

    try:
        has_mc  = "07_material_consumption.csv" in all_dfs
        has_po  = "03_production_orders.csv" in all_dfs
        has_sl  = "06_machine_shift_log.csv" in all_dfs

        if not (has_mc or has_po or has_sl):
            st.info("Upload files 03, 06, or 07 to enable trend analysis.")
        else:
            # ── Build weekly metrics dataframe ───────────────────────────────
            weekly_data = {}

            # 1. Material Efficiency % — from file 07
            if has_mc:
                df_mc = all_dfs["07_material_consumption.csv"].copy()
                df_mc["Issue_Date"] = pd.to_datetime(df_mc["Issue_Date"], errors="coerce")
                df_mc["Week"] = df_mc["Issue_Date"].dt.to_period("W").astype(str)
                df_mc["Material_Utilization_Pct"] = pd.to_numeric(
                    df_mc["Material_Utilization_Pct"], errors="coerce")
                weekly_data["Material Efficiency %"] = (
                    df_mc.groupby("Week")["Material_Utilization_Pct"].mean().reset_index()
                    .rename(columns={"Material_Utilization_Pct": "Value", "Week": "Period"})
                )

            # 2. Waste Cost INR — from file 07 + 01
            if has_mc and "01_raw_material_master.csv" in all_dfs:
                df_rm = all_dfs["01_raw_material_master.csv"].copy()
                df_mc2 = all_dfs["07_material_consumption.csv"].copy()
                df_mc2["Issue_Date"] = pd.to_datetime(df_mc2["Issue_Date"], errors="coerce")
                df_mc2["Week"] = df_mc2["Issue_Date"].dt.to_period("W").astype(str)
                df_mc2 = pd.merge(df_mc2, df_rm[["Material_Code","Unit_Cost_INR_per_kg"]],
                                  on="Material_Code", how="left")
                df_mc2["Unit_Cost_INR_per_kg"] = pd.to_numeric(
                    df_mc2["Unit_Cost_INR_per_kg"], errors="coerce")
                df_mc2["Scrap_Qty_kg"] = pd.to_numeric(df_mc2["Scrap_Qty_kg"], errors="coerce")
                df_mc2["Purge_Qty_kg"] = pd.to_numeric(df_mc2["Purge_Qty_kg"], errors="coerce")
                df_mc2["Waste_Cost"] = (df_mc2["Scrap_Qty_kg"] + df_mc2["Purge_Qty_kg"]) * df_mc2["Unit_Cost_INR_per_kg"]
                weekly_data["Waste Cost (INR)"] = (
                    df_mc2.groupby("Week")["Waste_Cost"].sum().reset_index()
                    .rename(columns={"Waste_Cost": "Value", "Week": "Period"})
                )

            # 3. Rejection Rate % — from file 03
            if has_po:
                df_po = all_dfs["03_production_orders.csv"].copy()
                df_po["Actual_Start"] = pd.to_datetime(df_po["Actual_Start"], errors="coerce")
                df_po["Week"] = df_po["Actual_Start"].dt.to_period("W").astype(str)
                df_po["Rejection_Rate_Pct"] = pd.to_numeric(
                    df_po["Rejection_Rate_Pct"], errors="coerce")
                weekly_data["Rejection Rate %"] = (
                    df_po.groupby("Week")["Rejection_Rate_Pct"].mean().reset_index()
                    .rename(columns={"Rejection_Rate_Pct": "Value", "Week": "Period"})
                )

            # 4. Machine Downtime — from file 06
            if has_sl:
                df_sl = all_dfs["06_machine_shift_log.csv"].copy()
                df_sl["Date"] = pd.to_datetime(df_sl["Date"], errors="coerce")
                df_sl["Week"] = df_sl["Date"].dt.to_period("W").astype(str)
                df_sl["Downtime_Min"] = pd.to_numeric(df_sl["Downtime_Min"], errors="coerce")
                weekly_data["Machine Downtime (mins)"] = (
                    df_sl.groupby("Week")["Downtime_Min"].sum().reset_index()
                    .rename(columns={"Downtime_Min": "Value", "Week": "Period"})
                )

            # ── Plot each metric ─────────────────────────────────────────────
            colors_map = {
                "Material Efficiency %":    "#10b981",
                "Waste Cost (INR)":         "#ef4444",
                "Rejection Rate %":         "#f59e0b",
                "Machine Downtime (mins)":  "#6366f1",
            }

            trend_insights = {}

            for metric, df_w in weekly_data.items():
                if df_w.empty or df_w["Value"].isna().all():
                    continue

                df_w = df_w.dropna(subset=["Value"]).sort_values("Period")
                if len(df_w) < 2:
                    continue

                # ── Chart ────────────────────────────────────────────────────
                fig_t = go.Figure()
                fig_t.add_trace(go.Bar(
                    x=df_w["Period"], y=df_w["Value"],
                    marker_color=colors_map.get(metric, "#2563eb"),
                    opacity=0.7, name=metric
                ))
                fig_t.add_trace(go.Scatter(
                    x=df_w["Period"], y=df_w["Value"],
                    mode="lines+markers",
                    line=dict(color=colors_map.get(metric, "#2563eb"), width=2.5),
                    marker=dict(size=8), name="Trend"
                ))

                # Week-on-week change arrow
                last_val  = df_w["Value"].iloc[-1]
                prev_val  = df_w["Value"].iloc[-2]
                change    = ((last_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                arrow     = "📈" if change > 0 else "📉"
                direction = "up" if change > 0 else "down"

                fig_t.update_layout(
                    title=f"{metric} — Weekly Trend  |  Last week: {arrow} {abs(change):.1f}% vs prior week",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(color="#cbd5e1"),
                    height=320,
                    margin=dict(l=20, r=20, t=50, b=60),
                    xaxis=dict(tickangle=-30, gridcolor="rgba(51,65,85,0.4)"),
                    yaxis=dict(gridcolor="rgba(51,65,85,0.4)"),
                    showlegend=False
                )
                st.plotly_chart(fig_t, use_container_width=True)

                # Store for AI insight
                trend_insights[metric] = {
                    "weeks":     df_w["Period"].tolist(),
                    "values":    df_w["Value"].round(2).tolist(),
                    "last":      round(last_val, 2),
                    "prev":      round(prev_val, 2),
                    "change_pct": round(change, 2),
                    "direction": direction
                }

            # ── AI Insight per metric ────────────────────────────────────────
            if trend_insights:
                st.markdown("---")
                st.markdown("### 🤖 AI Trend Insights")
                ai_key = st.session_state.get("user_api_key", "")

                if not ai_key:
                    st.info("🔑 Login as Admin or enter your API key in the Copilot section to get AI insights.")
                else:
                    if st.button("⚡ Generate AI Trend Insights", type="primary"):
                        import anthropic
                        client = anthropic.Anthropic(api_key=ai_key)

                        trend_text = ""
                        for metric, data in trend_insights.items():
                            trend_text += (
                                f"\n{metric}:\n"
                                f"  Weekly values: {data['values']}\n"
                                f"  Last week: {data['last']} | Prior week: {data['prev']} | "
                                f"Change: {data['change_pct']:+.1f}%\n"
                            )

                        prompt = (
                            "You are an operations analyst for a plastic injection moulding plant. "
                            "Analyse these week-on-week trends and give sharp, specific insights:\n"
                            + trend_text +
                            "\nFor each metric: 1 sentence on what happened, 1 sentence on why it might have happened, "
                            "1 sentence on what action to take. Be direct and specific. "
                            "Format as: **[Metric Name]**: insight. "
                            "End with one overall plant health verdict in bold."
                        )

                        with st.spinner("AI is analysing weekly trends..."):
                            response = client.messages.create(
                                model="claude-sonnet-4-5",
                                max_tokens=600,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            insight_text = response.content[0].text

                        st.markdown(insight_text)

    except Exception as e:
        st.warning(f"Trend analysis error: {e}")

# CUSTOM REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("")
with st.expander("🛠️ Custom Report Builder", expanded=False):
    st.markdown("Build your own visualizations from any uploaded dataset.")
    if all_dfs:
        selected_file = st.selectbox("Select Dataset:", list(all_dfs.keys()), key="custom_file")
        custom_df     = all_dfs[selected_file]
        all_columns   = custom_df.columns.tolist()
        numeric_columns = custom_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X Axis (Categories / Dates):", all_columns, index=0)
        with col2:
            y_axis = st.selectbox("Y Axis (Numeric):",
                                  numeric_columns if numeric_columns else all_columns,
                                  index=len(numeric_columns) - 1 if numeric_columns else 0)
        with col3:
            chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"])

        st.caption("Bar and Pie charts automatically aggregate (sum) Y values per X category.")

        if st.button("⚡ Generate Chart", type="primary"):
            try:
                if chart_type == "Bar Chart":
                    agg_df = custom_df.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.bar(agg_df, x=x_axis, y=y_axis, title=f"Total {y_axis} by {x_axis}",
                                 color_discrete_sequence=["#2563eb"])
                elif chart_type == "Line Chart":
                    agg_df = custom_df.groupby(x_axis)[y_axis].sum().reset_index().sort_values(by=x_axis)
                    fig = px.line(agg_df, x=x_axis, y=y_axis, markers=True,
                                  title=f"Trend of {y_axis} over {x_axis}",
                                  color_discrete_sequence=["#10b981"])
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(custom_df, x=x_axis, y=y_axis,
                                     title=f"Correlation: {x_axis} vs {y_axis}",
                                     color_discrete_sequence=["#f59e0b"])
                elif chart_type == "Pie Chart":
                    agg_df = custom_df.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.pie(agg_df, names=x_axis, values=y_axis,
                                 title=f"Distribution of {y_axis} by {x_axis}",
                                 color_discrete_sequence=px.colors.sequential.Blues_r)

                if chart_type != "Pie Chart":
                    fig.update_xaxes(tickangle=-30, gridcolor='rgba(51,65,85,0.4)')
                    fig.update_yaxes(gridcolor='rgba(51,65,85,0.4)')

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(color="#cbd5e1"),
                    margin=dict(l=20, r=20, t=45, b=80),
                    height=480,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Could not generate chart. Error: {e}")
    else:
        st.info("Upload data to use the Custom Report Builder.")

# ══════════════════════════════════════════════════════════════════════════════
# MORNING REPORT — AI AUTO-WRITES + PDF EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📄 Morning Operations Report</div>', unsafe_allow_html=True)
st.markdown("Click below — the AI will analyse your data, write a summary, and generate a PDF report ready for your morning meeting.")

report_api_key = st.session_state.get("user_api_key", "") or st.secrets.get("ANTHROPIC_API_KEY", "")

rpt_col1, rpt_col2 = st.columns([2, 5])
with rpt_col1:
    generate_report = st.button("⚡ Generate Morning Report", type="primary", use_container_width=True)

if generate_report:
    if not report_api_key:
        report_api_key = st.session_state.get("manual_api_key","")
    if not report_api_key:
        st.warning("Enter your API key in the Copilot section below first.")
    else:
        try:
            import anthropic
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                            Table, TableStyle, HRFlowable, Image as RLImage)
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            import plotly.io as pio

            with st.spinner("AI is analysing your data and writing the report..."):

                # 1. Collect KPIs
                kpi_summary = {}
                alert_list  = []
                req = ["07_material_consumption.csv","01_raw_material_master.csv",
                       "03_production_orders.csv","09_raw_material_inventory.csv"]

                if all(f in all_dfs for f in req):
                    df_mc  = all_dfs["07_material_consumption.csv"]
                    df_rm  = all_dfs["01_raw_material_master.csv"]
                    df_po  = all_dfs["03_production_orders.csv"]
                    df_inv = all_dfs["09_raw_material_inventory.csv"]
                    avg_eff = df_mc["Material_Utilization_Pct"].mean()
                    df_mc_c = pd.merge(df_mc, df_rm[["Material_Code","Unit_Cost_INR_per_kg"]],
                                       on="Material_Code", how="left")
                    df_mc_c["Waste_Cost_INR"] = ((df_mc_c["Scrap_Qty_kg"] + df_mc_c["Purge_Qty_kg"])
                                                  * df_mc_c["Unit_Cost_INR_per_kg"])
                    waste_cost  = df_mc_c["Waste_Cost_INR"].sum()
                    total_wos   = len(df_po["WO_ID"].unique())
                    df_inv_a    = pd.merge(df_inv, df_rm[["Material_Code","MOQ_kg"]],
                                           on="Material_Code", how="left")
                    crit_alerts = len(df_inv_a[df_inv_a["Qty_Available_kg"] < df_inv_a["MOQ_kg"]])
                    kpi_summary = {
                        "Avg Material Efficiency":   f"{avg_eff:.1f}%",
                        "Total Waste Cost (INR)":    f"Rs {waste_cost:,.0f}",
                        "Processed Work Orders":     str(total_wos),
                        "Critical Inventory Alerts": str(crit_alerts),
                    }
                    low_stock = df_inv_a[df_inv_a["Qty_Available_kg"] < df_inv_a["MOQ_kg"]]
                    for _, row in low_stock.head(4).iterrows():
                        alert_list.append(f"LOW STOCK: {row['Material_Code']} — "
                                          f"{row['Qty_Available_kg']:.1f} kg available (MOQ: {row['MOQ_kg']:.1f} kg)")
                    top_waste = df_mc_c.groupby("Material_Code")["Waste_Cost_INR"].sum().nlargest(2)
                    for mat, cost in top_waste.items():
                        alert_list.append(f"HIGH WASTE COST: {mat} — Rs {cost:,.0f}")

                if "06_machine_shift_log.csv" in all_dfs:
                    df_shift = all_dfs["06_machine_shift_log.csv"]
                    top_def  = df_shift.groupby("Rejection_Reason")["Rejected_Parts"].sum().idxmax()
                    top_def_n = df_shift.groupby("Rejection_Reason")["Rejected_Parts"].sum().max()
                    alert_list.append(f"TOP DEFECT: {top_def} — {top_def_n:,.0f} rejected parts")

                # 2. Ask Claude to write the executive summary
                kpi_lines   = "\n".join([f"- {k}: {v}" for k, v in kpi_summary.items()])
                alert_lines = "\n".join([f"- {a}" for a in alert_list]) if alert_list else "- None"
                data_ctx = (
                    "You are writing a morning operations report for a plastic injection moulding plant.\n\n"
                    f"KPIs:\n{kpi_lines}\n\n"
                    f"Alerts:\n{alert_lines}\n\n"
                    "Write a concise professional executive summary (4-5 sentences) for a morning meeting. "
                    "Include overall plant health, top concern, and one clear recommendation. "
                    "No bullet points. End with one bold action item for the shift manager."
                )
                client   = anthropic.Anthropic(api_key=report_api_key)
                response = client.messages.create(model="claude-sonnet-4-5", max_tokens=400,
                                                  messages=[{"role":"user","content":data_ctx}])
                ai_summary = response.content[0].text

                # 3. Generate chart images
                chart_images = []
                if "06_machine_shift_log.csv" in all_dfs:
                    df_s2 = all_dfs["06_machine_shift_log.csv"]
                    ddf   = df_s2.groupby("Rejection_Reason")["Rejected_Parts"].sum().reset_index()
                    ddf   = ddf.sort_values("Rejected_Parts", ascending=False)
                    ddf["Cum"] = ddf["Rejected_Parts"].cumsum() / ddf["Rejected_Parts"].sum() * 100
                    fp = go.Figure()
                    fp.add_trace(go.Bar(x=ddf["Rejection_Reason"], y=ddf["Rejected_Parts"],
                                        marker_color="#f59e0b"))
                    fp.add_trace(go.Scatter(x=ddf["Rejection_Reason"], y=ddf["Cum"],
                                            yaxis="y2", mode="lines+markers",
                                            line=dict(color="#ef4444", width=2)))
                    fp.update_layout(title="Scrap Pareto Analysis", height=340, width=720,
                                     paper_bgcolor="white", plot_bgcolor="#f8fafc",
                                     font=dict(color="#1e293b"), showlegend=False,
                                     margin=dict(l=40,r=40,t=50,b=60),
                                     yaxis2=dict(overlaying="y", side="right", range=[0,105]))
                    chart_images.append(("Scrap Pareto Analysis (80/20 Rule)",
                                         pio.to_image(fp, format="png", scale=2)))

                if all(f in all_dfs for f in ["07_material_consumption.csv",
                                               "03_production_orders.csv","02_part_master.csv"]):
                    df_mc2 = all_dfs["07_material_consumption.csv"]
                    df_po2 = all_dfs["03_production_orders.csv"]
                    df_b2  = all_dfs["02_part_master.csv"]
                    df_pb  = pd.merge(df_po2, df_b2[["Part_Code","Material_per_Part_g"]],
                                      on="Part_Code", how="left")
                    df_pb["Theoretical_Good_kg"] = df_pb["Actual_Good_Parts"] * df_pb["Material_per_Part_g"] / 1000
                    df_fl  = pd.merge(df_mc2, df_pb[["WO_ID","Theoretical_Good_kg"]], on="WO_ID", how="left")
                    ti = df_fl["Qty_Issued_kg"].sum()
                    tr = df_fl["Returned_to_Store_kg"].sum()
                    tp = ti - tr
                    ts = df_fl["Scrap_Qty_kg"].sum()
                    tpu= df_fl["Purge_Qty_kg"].sum()
                    tg = df_fl["Theoretical_Good_kg"].sum()
                    tgh= tp - (tg + ts + tpu)
                    fm = go.Figure(go.Bar(
                        x=["Good Parts","Logged Scrap","Purge Waste","Ghost Leak"],
                        y=[tg, ts, tpu, max(tgh,0)],
                        marker_color=["#10b981","#f59e0b","#f97316","#ef4444"],
                        text=[f"{v:,.0f} kg" for v in [tg,ts,tpu,max(tgh,0)]],
                        textposition="outside"
                    ))
                    fm.update_layout(title="Material Mass Balance Summary", height=320, width=720,
                                     paper_bgcolor="white", plot_bgcolor="#f8fafc",
                                     font=dict(color="#1e293b"), showlegend=False,
                                     margin=dict(l=40,r=40,t=50,b=40), yaxis=dict(title="kg"))
                    chart_images.append(("Material Mass Balance",
                                         pio.to_image(fm, format="png", scale=2)))

                # 4. Build PDF
                pdf_buf = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buf, pagesize=A4,
                                        leftMargin=2*cm, rightMargin=2*cm,
                                        topMargin=2*cm, bottomMargin=2*cm)
                styles = getSampleStyleSheet()
                elems  = []

                title_s   = ParagraphStyle("T", fontSize=20, fontName="Helvetica-Bold",
                                           textColor=rl_colors.HexColor("#1e3a8a"), spaceAfter=4)
                sub_s     = ParagraphStyle("S", fontSize=9,  fontName="Helvetica",
                                           textColor=rl_colors.HexColor("#64748b"), spaceAfter=2)
                sec_s     = ParagraphStyle("SE",fontSize=12, fontName="Helvetica-Bold",
                                           textColor=rl_colors.HexColor("#1e3a8a"),
                                           spaceBefore=12, spaceAfter=6)
                body_s    = ParagraphStyle("B", fontSize=9.5,fontName="Helvetica",
                                           textColor=rl_colors.HexColor("#334155"), leading=15)
                alert_s   = ParagraphStyle("A", fontSize=9,  fontName="Helvetica",
                                           textColor=rl_colors.HexColor("#92400e"), leading=13)
                footer_s  = ParagraphStyle("F", fontSize=7.5,fontName="Helvetica",
                                           textColor=rl_colors.HexColor("#94a3b8"),
                                           alignment=TA_CENTER)

                elems.append(Paragraph("Raw Material Optimization", title_s))
                elems.append(Paragraph(
                    f"Morning Operations Report  |  {pd.Timestamp.now().strftime('%d %b %Y, %I:%M %p')}",
                    sub_s))
                elems.append(HRFlowable(width="100%", thickness=2,
                                         color=rl_colors.HexColor("#1e3a8a"), spaceAfter=12))

                if kpi_summary:
                    elems.append(Paragraph("Plant Health Metrics", sec_s))
                    kd = [["Metric","Value"]] + [[k,v] for k,v in kpi_summary.items()]
                    kt = Table(kd, colWidths=[10*cm, 6*cm])
                    kt.setStyle(TableStyle([
                        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1e3a8a")),
                        ("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
                        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                        ("FONTSIZE",(0,0),(-1,0),10),
                        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
                        ("FONTSIZE",(0,1),(-1,-1),9.5),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),
                         [rl_colors.HexColor("#f0f7ff"),rl_colors.white]),
                        ("TEXTCOLOR",(0,1),(-1,-1),rl_colors.HexColor("#1e293b")),
                        ("GRID",(0,0),(-1,-1),0.5,rl_colors.HexColor("#cbd5e1")),
                        ("ROWHEIGHT",(0,0),(-1,-1),22),
                        ("LEFTPADDING",(0,0),(-1,-1),10),
                        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                    ]))
                    elems.append(kt)
                    elems.append(Spacer(1,10))

                elems.append(Paragraph("AI Executive Summary", sec_s))
                elems.append(HRFlowable(width="100%", thickness=0.5,
                                         color=rl_colors.HexColor("#cbd5e1"), spaceAfter=6))
                for para in ai_summary.split("\n"):
                    if para.strip():
                        elems.append(Paragraph(para.strip(), body_s))
                elems.append(Spacer(1,8))

                if alert_list:
                    elems.append(Paragraph("Active Alerts", sec_s))
                    ad = [[Paragraph(f"  {a}", alert_s)] for a in alert_list]
                    at = Table(ad, colWidths=[16*cm])
                    at.setStyle(TableStyle([
                        ("ROWBACKGROUNDS",(0,0),(-1,-1),
                         [rl_colors.HexColor("#fef3c7"),rl_colors.HexColor("#fffbeb")]),
                        ("GRID",(0,0),(-1,-1),0.3,rl_colors.HexColor("#fcd34d")),
                        ("LEFTPADDING",(0,0),(-1,-1),10),
                        ("TOPPADDING",(0,0),(-1,-1),6),
                        ("BOTTOMPADDING",(0,0),(-1,-1),6),
                    ]))
                    elems.append(at)
                    elems.append(Spacer(1,10))

                if chart_images:
                    elems.append(Paragraph("Visual Analytics", sec_s))
                    elems.append(HRFlowable(width="100%", thickness=0.5,
                                             color=rl_colors.HexColor("#cbd5e1"), spaceAfter=8))
                    for ctitle, ibytes in chart_images:
                        elems.append(Paragraph(ctitle, ParagraphStyle("CT", fontSize=10,
                                                fontName="Helvetica-Bold",
                                                textColor=rl_colors.HexColor("#475569"),
                                                spaceAfter=4)))
                        elems.append(RLImage(io.BytesIO(ibytes), width=16*cm, height=7*cm))
                        elems.append(Spacer(1,12))

                elems.append(HRFlowable(width="100%", thickness=0.5,
                                         color=rl_colors.HexColor("#cbd5e1"), spaceBefore=10))
                elems.append(Paragraph(
                    "Generated by Ops Intelligence Agent  |  Powered by Claude AI  |  Confidential",
                    footer_s))

                doc.build(elems)
                pdf_buf.seek(0)

            st.success("Morning report is ready!")
            st.markdown("**AI Executive Summary:**")
            st.info(ai_summary)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buf.getvalue(),
                file_name=f"Morning_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary"
            )

        except ImportError as ie:
            st.error(f"Missing library: {ie}. Run: pip install reportlab anthropic kaleido pillow")
        except Exception as e:
            st.error(f"Report generation failed: {e}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# AI COPILOT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🤖 Claude Operations Copilot</div>', unsafe_allow_html=True)

# ── Admin vs User login ─────────────────────────────────────────────────────
ADMIN_PASSWORD  = st.secrets.get("ADMIN_PASSWORD", "ops@admin2024")
ADMIN_API_KEY   = st.secrets.get("ANTHROPIC_API_KEY", "")

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

login_col1, login_col2 = st.columns([3, 2])

with login_col1:
    if not st.session_state["is_admin"]:
        st.markdown("**🔑 API Key Access**")
        st.caption("Admin: enter admin password. Others: enter your own Anthropic API key.")

        access_mode = st.radio("Access Mode", ["👤 Enter my own API key", "🔐 Admin login"],
                                horizontal=True, key="access_mode")

        if access_mode == "🔐 Admin login":
            admin_pwd = st.text_input("Admin Password", type="password", key="admin_pwd")
            if st.button("Login as Admin", type="primary"):
                if admin_pwd == ADMIN_PASSWORD:
                    st.session_state["is_admin"]    = True
                    st.session_state["user_api_key"] = ADMIN_API_KEY
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
        else:
            api_key_input = st.text_input(
                "Your Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                value=st.session_state.get("user_api_key", ""),
                help="Get your key at console.anthropic.com"
            )
            if api_key_input:
                st.session_state["user_api_key"] = api_key_input
    else:
        st.success("🔐 Logged in as **Admin** — API key loaded automatically.")
        if st.button("Logout"):
            st.session_state["is_admin"]     = False
            st.session_state["user_api_key"] = ""
            st.rerun()

with login_col2:
    if st.session_state["is_admin"]:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e3a8a,#1d4ed8);
        border-radius:10px;padding:1rem;text-align:center;margin-top:1.5rem'>
        <div style='color:#93c5fd;font-size:0.75rem;font-weight:600;
        text-transform:uppercase;letter-spacing:1px'>Admin Access</div>
        <div style='color:white;font-size:1.5rem'>✅</div>
        <div style='color:#bfdbfe;font-size:0.8rem'>Full access enabled</div>
        </div>""", unsafe_allow_html=True)

api_key = st.session_state.get("user_api_key", "")
if api_key:
    os.environ["ANTHROPIC_API_KEY"] = api_key

    file_options = list(all_dfs.keys())
    if len(file_options) > 1:
        file_options.append("🔗 All Files Combined")

    selected = st.selectbox(
        "Ask questions about:",
        file_options,
        help="Select which file the AI should analyse."
    )

    if selected == "🔗 All Files Combined":
        combined_frames = []
        for fname, frame in all_dfs.items():
            frame = frame.copy()
            frame['__source_file__'] = fname
            combined_frames.append(frame)
        agent_df   = pd.concat(combined_frames, ignore_index=True, sort=False)
        file_context = (
            f"The dataframe is a combined view of {len(all_dfs)} files: {', '.join(all_dfs.keys())}. "
            f"A column called '__source_file__' tells you which file each row came from. "
            f"Total rows: {len(agent_df):,}. Columns: {', '.join(agent_df.columns.tolist())}."
        )
    else:
        agent_df   = all_dfs[selected]
        file_context = (
            f"The dataframe is from the file '{selected}'. "
            f"Rows: {len(agent_df):,}. Columns: {', '.join(agent_df.columns.tolist())}."
        )

    system_hint = (
        "You are an expert operations and manufacturing analyst specialising in "
        "raw material optimization, plastic injection moulding, and supply chain. "
        f"{file_context} "
        "Always answer with specific numbers, INR costs where relevant, percentages, "
        "and clear actionable recommendations. If asked to compare, use the data directly. "
        "Be concise and structured in your responses."
    )

    llm   = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm, agent_df, verbose=True,
        allow_dangerous_code=True, prefix=system_hint
    )

    # Suggested questions
    st.markdown('<div class="suggestion-label">💡 Quick Questions</div>', unsafe_allow_html=True)
    suggestions = [
        "Which material has the highest scrap quantity?",
        "What is the overall rejection rate?",
        "Which machine has the most downtime?",
        "Show total material consumption by polymer type",
        "Which customer has the most pending orders?",
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_prompt = suggestion

    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = None
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")
    else:
        prompt = st.chat_input("Ask anything about your operations data…")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Analysing your data…"):
                try:
                    response = agent.invoke(prompt)
                    answer   = response["output"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

else:
    st.info("🔑 Enter your Anthropic API key above to activate the AI Copilot.")
