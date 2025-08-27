

from __future__ import annotations

import io
import os
import sys
import json
import math
import textwrap
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st

# ML
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, classification_report

# Charts
import plotly.express as px
import plotly.graph_objects as go

# Excel writer
from io import BytesIO

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="OSG Warranty Intelligence Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Helper Utilities
# ---------------------------

def _coerce_str(x) -> str:
    """Safely convert serial/IMEI-like values to string without scientific notation or rounding.
    Keeps leading zeros and strips whitespace."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Fix Excel scientific notation like 9.12345E+15
    if "e+" in s.lower():
        try:
            as_int = int(float(s))
            return f"{as_int:d}"
        except Exception:
            pass
    # If it's numeric but with .0 at end, drop decimals
    if s.endswith('.0') and s.replace('.', '', 1).isdigit():
        return s[:-2]
    return s


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def parse_date(s: pd.Series, dayfirst=True) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns}
    return df.rename(columns=mapping)


def safe_div(a, b):
    return (a / b) if (b not in [0, None, np.nan] and b != 0) else 0


def money_fmt(x: float) -> str:
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return str(x)


def mask_phone(s: str) -> str:
    s = _coerce_str(s)
    return ("*" * max(0, len(s) - 4)) + s[-4:]


def read_any(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        elif name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded)
        else:
            st.warning(f"Unsupported file type: {uploaded.name}")
            return None
        return normalize_colnames(df)
    except Exception as e:
        st.error(f"Failed to read {uploaded.name}: {e}")
        return None


# ---------------------------
# Domain-Specific Cleaning
# ---------------------------

def standardize_osg_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Standardize common OSG column names across sales/warranty/claims/master.
    The function is resilient to missing columns; it creates them with defaults if absent."""
    if df is None:
        return pd.DataFrame()

    # Canonical names we expect to work with
    canonical = [
        "customer_mobile","date","invoice_number","customer_name","store_code","branch","region",
        "imei","serial_no","category","brand","model","quantity","item_rate","plan_price","sold_price",
        "plan_type","ews_qty","email","manufacturer_warranty","retailer_sku","onsitego_sku","duration_year",
        "total_coverage","comment","return_flag","return_against_invoice_no","primary_invoice_no",
        "rbm","bdm","staff"
    ]

    # Friendly aliases seen in wild data
    aliases: Dict[str, List[str]] = {
        "customer_mobile": ["mobile", "customer_mobile_no", "customer_phone", "phone"],
        "date": ["bill_date", "txn_date", "invoice_date"],
        "invoice_number": ["invoice_no", "bill_no", "invno"],
        "imei": ["imei_1", "imei_no", "imei number"],
        "serial_no": ["serial", "sn", "serialnumber", "serial_no."],
        "plan_price": ["mrp", "plan_mrp"],
        "sold_price": ["amount", "net_amount", "netamt", "plan_amount"],
        "duration_year": ["duration", "duration_(year)", "warranty_duration"],
        "ews_qty": ["ews_qty."],
        "manufacturer_warranty": ["oem_warranty", "mfr_warranty"],
        "retailer_sku": ["retailer_sku_code", "retailer_item_code"],
        "onsitego_sku": ["onsitego_item_code", "osg_sku"],
        "total_coverage": ["coverage", "total_coverage_value"],
        "return_flag": ["is_return", "return"],
        "return_against_invoice_no": ["return_invoice", "return_against"],
        "primary_invoice_no": ["parent_invoice", "original_invoice"],
        "rbm": ["rbm_name"],
        "bdm": ["bdm_name"],
        "staff": ["salesman", "executive", "staff_name"],
    }

    # Promote aliases
    for target, alist in aliases.items():
        if target not in df.columns:
            for a in alist:
                if a in df.columns:
                    df[target] = df[a]
                    break

    # Ensure all expected columns exist
    for c in canonical:
        if c not in df.columns:
            df[c] = np.nan

    # Type fixes
    df['date'] = parse_date(df['date'])
    for c in ['item_rate','plan_price','sold_price','total_coverage']:
        df[c] = to_numeric(df[c])
    for c in ['imei','serial_no','customer_mobile','invoice_number','primary_invoice_no', 'return_against_invoice_no']:
        df[c] = df[c].apply(_coerce_str)

    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

    # Standardize brand/category/model strings
    for c in ['brand','category','model','branch','region','rbm','bdm','staff','store_code']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper().replace({'NAN':'', 'NONE':''})

    # Duration normalize to float years
    if 'duration_year' in df.columns:
        df['duration_year'] = pd.to_numeric(df['duration_year'], errors='coerce').fillna(0)

    # Derived
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    df['dow'] = df['date'].dt.dayofweek

    # Flags
    df['is_warranty_line'] = (df['plan_price'] > 0) | (df['onsitego_sku'].astype(str).str.len() > 0)

    # Primary keys
    df['line_key'] = df['invoice_number'].astype(str) + '|' + df['onsitego_sku'].astype(str) + '|' + df['imei'].astype(str)

    df['_kind'] = kind
    return df


def merge_sales_warranty(sales: pd.DataFrame, warranty: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sales_enriched, warranty_enriched) with linking keys.
    Link order: invoice_number -> (customer_mobile, date window) -> (IMEI/Serial)."""
    if sales is None or len(sales)==0:
        return pd.DataFrame(), pd.DataFrame()

    if warranty is None:
        warranty = pd.DataFrame(columns=sales.columns)

    s = sales.copy()
    w = warranty.copy()

    # First: invoice number exact match
    w['link_key'] = w['invoice_number']
    s['link_key'] = s['invoice_number']

    linked = w.merge(s[['invoice_number','customer_mobile','date','store_code','model','brand','category','item_rate','sold_price','line_key']].rename(columns={'line_key':'sale_line_key'}),
                     left_on='link_key', right_on='invoice_number', how='left', suffixes=("_w","_s"))

    # Fallback join by (customer_mobile, +/-3 days)
    nohit = linked['sale_line_key'].isna()
    if nohit.any():
        w2 = w.loc[nohit, ['customer_mobile','date','model','brand','category','invoice_number','line_key']].copy()
        s2 = s[['customer_mobile','date','model','brand','category','invoice_number','line_key']].copy()
        # Cartesian within 3-day window same mobile
        cand = w2.merge(s2, on=['customer_mobile'], how='left', suffixes=("_w","_s"))
        cand['d_days'] = (cand['date_w'] - cand['date_s']).abs().dt.days
        cand = cand[cand['d_days'] <= 3]
        # Prefer same model/brand/category if available
        cand['score'] = 0
        cand.loc[cand['model_w'] == cand['model_s'], 'score'] += 2
        cand.loc[cand['brand_w'] == cand['brand_s'], 'score'] += 1
        cand.loc[cand['category_w'] == cand['category_s'], 'score'] += 1
        # pick best match per w.line_key
        cand = cand.sort_values(['line_key_w','score','d_days'], ascending=[True, False, True])
        best = cand.groupby('line_key_w').head(1)
        # Map back
        map_sale_key = dict(zip(best['line_key_w'], best['line_key_s']))
        linked.loc[nohit, 'sale_line_key'] = linked.loc[nohit, 'line_key'].map(map_sale_key)

    # Another fallback by IMEI/Serial
    nohit = linked['sale_line_key'].isna()
    if nohit.any():
        key_by_imei = s.set_index('imei')['line_key'].to_dict()
        linked.loc[nohit, 'sale_line_key'] = linked.loc[nohit, 'imei'].map(key_by_imei)

    # Build indicators on sales
    s['warranty_attached'] = s['line_key'].isin(linked['sale_line_key'].dropna().unique())

    return s, linked


# ---------------------------
# KPI Calculations
# ---------------------------

def compute_kpis(sales: pd.DataFrame, warranty_linked: pd.DataFrame) -> Dict[str, float]:
    if sales is None or len(sales)==0:
        return {}
    total_sales = len(sales)
    with_warranty = int(sales['warranty_attached'].sum())
    conversion = safe_div(with_warranty, total_sales)

    rev_plan = to_numeric(warranty_linked['plan_price']).sum()
    rev_sold = to_numeric(warranty_linked['sold_price']).sum()

    # Missed revenue: estimate avg plan price per (brand, category)
    avg_plan = warranty_linked.groupby(['brand','category'])['sold_price'].mean().rename('avg_plan').reset_index()
    sales_wo = sales.loc[~sales['warranty_attached']]
    sales_wo = sales_wo.merge(avg_plan, on=['brand','category'], how='left')
    sales_wo['avg_plan'] = sales_wo['avg_plan'].fillna(warranty_linked['sold_price'].mean())
    missed_rev = to_numeric(sales_wo['avg_plan']).sum()

    return {
        'total_sales_lines': int(total_sales),
        'warranty_attached_lines': int(with_warranty),
        'conversion_rate_pct': round(conversion*100, 2),
        'warranty_revenue_sold': float(rev_sold),
        'warranty_revenue_mrp': float(rev_plan),
        'missed_revenue_estimate': float(missed_rev),
    }


def group_conversion(sales: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    if sales is None or len(sales)==0:
        return pd.DataFrame()
    g = sales.groupby(by).agg(
        sales_lines=('line_key','count'),
        with_warranty=('warranty_attached','sum')
    ).reset_index()
    g['conversion_pct'] = g.apply(lambda r: safe_div(r['with_warranty'], r['sales_lines'])*100, axis=1)
    g = g.sort_values('conversion_pct', ascending=False)
    return g


# ---------------------------
# ML: Next-Best-Offer (Warranty Propensity)
# ---------------------------

def build_propensity_model(sales: pd.DataFrame) -> Tuple[Optional[Pipeline], Optional[float]]:
    df = sales.copy()
    if 'warranty_attached' not in df.columns or df['warranty_attached'].isna().all():
        return None, None

    # Features
    df['price_gap'] = df['item_rate'] - df['sold_price']
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)

    num_cols = ['item_rate','sold_price','price_gap']
    cat_cols = ['brand','category','model','store_code','rbm','bdm','staff','ym','dow']

    df = df.fillna({c: 'UNK' for c in cat_cols}).fillna({c: 0 for c in num_cols})

    X = df[num_cols + cat_cols]
    y = df['warranty_attached'].astype(int)

    pre = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([('pre', pre), ('gb', clf)])

    try:
        pipe.fit(X, y)
        # AUC using in-sample as a proxy (if no split provided)
        pred = pipe.predict_proba(X)[:,1]
        auc = roc_auc_score(y, pred)
        return pipe, float(auc)
    except Exception as e:
        st.warning(f"Propensity model training failed: {e}")
        return None, None


def score_propensity(pipe: Pipeline, sales: pd.DataFrame) -> pd.DataFrame:
    if pipe is None or sales is None or len(sales)==0:
        return pd.DataFrame()
    df = sales.copy()
    df['price_gap'] = df['item_rate'] - df['sold_price']
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)

    num_cols = ['item_rate','sold_price','price_gap']
    cat_cols = ['brand','category','model','store_code','rbm','bdm','staff','ym','dow']

    df = df.fillna({c: 'UNK' for c in cat_cols}).fillna({c: 0 for c in num_cols})
    X = df[num_cols + cat_cols]

    try:
        df['propensity'] = pipe.predict_proba(X)[:,1]
        return df
    except Exception as e:
        st.warning(f"Scoring failed: {e}")
        return pd.DataFrame()


# ---------------------------
# ML: Time-Series Forecast (Store x Month)
# ---------------------------

def build_forecast(df_w: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Pipeline]]:
    """Aggregate warranty sold count by store x month and fit a simple regression with lag features.
    Returns (forecast_df, model)."""
    if df_w is None or len(df_w)==0:
        return pd.DataFrame(), None
    w = df_w.copy()
    w['ym'] = w['date'].dt.to_period('M').astype(str)
    agg = w.groupby(['store_code','ym']).size().rename('w_cnt').reset_index()

    # Create time index
    all_months = pd.period_range(agg['ym'].min(), agg['ym'].max(), freq='M').astype(str)
    stores = agg['store_code'].dropna().unique().tolist()

    frames = []
    for sc in stores:
        a = agg[agg['store_code']==sc].set_index('ym').reindex(all_months).fillna(0).rename_axis('ym').reset_index()
        a['store_code'] = sc
        frames.append(a)
    panel = pd.concat(frames, ignore_index=True)

    # Lag features
    panel['w_cnt_lag1'] = panel.groupby('store_code')['w_cnt'].shift(1)
    panel['w_cnt_lag2'] = panel.groupby('store_code')['w_cnt'].shift(2)
    panel['w_cnt_ma3'] = panel.groupby('store_code')['w_cnt'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    panel = panel.fillna(0)

    # Train a simple regressor on known months; then project next 2 months
    # Encode store_code
    pre = ColumnTransformer([
        ('num', 'passthrough', ['w_cnt_lag1','w_cnt_lag2','w_cnt_ma3']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['store_code'])
    ])
    reg = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([('pre', pre), ('gb', reg)])

    X = panel[['store_code','w_cnt_lag1','w_cnt_lag2','w_cnt_ma3']]
    y = panel['w_cnt']
    try:
        pipe.fit(X, y)
    except Exception as e:
        st.warning(f"Forecast training failed: {e}")
        return pd.DataFrame(), None

    # Forecast next 2 months sequentially
    last_month = pd.Period(panel['ym'].max(), freq='M')
    future_months = [(last_month + i).strftime('%Y-%m') for i in [1,2]]

    forecasts = []
    base = panel.copy()
    for m in future_months:
        feat = []
        for sc in stores:
            sub = base[base['store_code']==sc].sort_values('ym')
            lag1 = sub['w_cnt'].iloc[-1] if len(sub)>0 else 0
            lag2 = sub['w_cnt'].iloc[-2] if len(sub)>1 else 0
            ma3 = sub['w_cnt'].iloc[-3:].mean() if len(sub)>0 else 0
            feat.append({'store_code':sc,'ym':m,'w_cnt_lag1':lag1,'w_cnt_lag2':lag2,'w_cnt_ma3':ma3})
        feat = pd.DataFrame(feat)
        yhat = pipe.predict(feat[['store_code','w_cnt_lag1','w_cnt_lag2','w_cnt_ma3']])
        feat['w_cnt'] = np.maximum(0, np.round(yhat,0))
        forecasts.append(feat)
        # Append to base for next step lags
        base = pd.concat([base, feat[['store_code','ym','w_cnt']]], ignore_index=True)

    fdf = pd.concat([panel[['store_code','ym','w_cnt']], *[f[['store_code','ym','w_cnt']] for f in forecasts]], ignore_index=True)
    fdf['is_forecast'] = ~fdf['ym'].isin(panel['ym'].unique())
    return fdf, pipe


# ---------------------------
# Anomaly Detection
# ---------------------------

def detect_anomalies(warranty: pd.DataFrame) -> pd.DataFrame:
    if warranty is None or len(warranty)==0:
        return pd.DataFrame()
    df = warranty.copy()
    df['discount_pct'] = np.where(df['plan_price']>0, (df['plan_price']-df['sold_price'])/df['plan_price'], 0)
    feats = df[['sold_price','plan_price','discount_pct']].fillna(0)
    try:
        iso = IsolationForest(random_state=42, contamination=0.03)
        df['anomaly_score'] = iso.fit_predict(feats)  # -1 anomalous
        df['is_anomaly'] = (df['anomaly_score'] == -1)
        return df
    except Exception as e:
        st.warning(f"Anomaly detection failed: {e}")
        return pd.DataFrame()


# ---------------------------
# Excel Export
# ---------------------------

def build_excel_report(kpis: Dict[str,float], convs: Dict[str,pd.DataFrame],
                       sales_scored: pd.DataFrame, warranty_linked: pd.DataFrame,
                       anomalies: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # KPI sheet
        kdf = pd.DataFrame([kpis])
        kdf.to_excel(writer, index=False, sheet_name='KPIs')

        for name, df in convs.items():
            if df is None or len(df)==0:
                continue
            df.to_excel(writer, index=False, sheet_name=f'Conv_{name[:22]}')

        if sales_scored is not None and len(sales_scored)>0:
            cols = ['invoice_number','date','customer_mobile','store_code','brand','category','model','item_rate','sold_price','rbm','bdm','staff','propensity']
            sales_scored[cols].sort_values('propensity', ascending=False).to_excel(writer, index=False, sheet_name='NextBestOffer')

        if warranty_linked is not None and len(warranty_linked)>0:
            wcols = ['invoice_number','date','store_code','brand','category','model','plan_price','sold_price','rbm','bdm','staff']
            warranty_linked[wcols].to_excel(writer, index=False, sheet_name='WarrantyLines')

        if anomalies is not None and len(anomalies)>0:
            acols = ['invoice_number','date','store_code','brand','model','plan_price','sold_price','discount_pct','is_anomaly']
            anomalies[acols].sort_values('discount_pct', ascending=False).to_excel(writer, index=False, sheet_name='Anomalies')

        # Formatting
        try:
            wb  = writer.book
            for sname in writer.sheets:
                ws = writer.sheets[sname]
                ws.set_column(0, 20, 16)
            # Conditional format on conversion sheets
            for name in convs.keys():
                sname = f'Conv_{name[:22]}'
                if sname in writer.sheets:
                    ws = writer.sheets[sname]
                    # conversion_pct assumed at col index where it appears
                    # Apply a color scale (green high, red low)
                    last_col = convs[name].shape[1]-1
                    ws.conditional_format(1, last_col, 10000, last_col, {
                        'type': '3_color_scale',
                        'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B'
                    })
        except Exception:
            pass

    return output.getvalue()


# ---------------------------
# UI: Sidebar - Uploads & Controls
# ---------------------------
with st.sidebar:
    st.title("🛡️ OSG Warranty Intelligence")
    st.caption("Deep analysis • ML • Forecast • Gamification • Renewals")

    st.subheader("Data Uploads")
    up_sales = st.file_uploader("Sales file (CSV/XLSX)", type=["csv","xls","xlsx"], key="sales")
    up_warranty = st.file_uploader("Warranty file (CSV/XLSX)", type=["csv","xls","xlsx"], key="warranty")
    up_claims = st.file_uploader("Claims file (optional)", type=["csv","xls","xlsx"], key="claims")
    up_store = st.file_uploader("Store master (optional)", type=["csv","xls","xlsx"], key="store")
    up_staff = st.file_uploader("RBM/BDM/Staff map (optional)", type=["csv","xls","xlsx"], key="staff")

    st.divider()
    st.subheader("Global Filters")
    flt_from = st.date_input("From date", value=dt.date.today() - dt.timedelta(days=120))
    flt_to   = st.date_input("To date", value=dt.date.today())
    region_sel = st.text_input("Region filter (comma-separated)")

    st.divider()
    run_models = st.checkbox("Run ML models (Propensity, Forecast)", value=True)
    show_phone_mask = st.checkbox("Mask phone numbers in UI", value=True)


# ---------------------------
# Load & Prepare Data
# ---------------------------
sales_raw = read_any(up_sales)
warranty_raw = read_any(up_warranty)
claims_raw = read_any(up_claims)
store_raw = read_any(up_store)
staff_raw = read_any(up_staff)

sales = standardize_osg_columns(sales_raw, kind='SALES')
warranty = standardize_osg_columns(warranty_raw, kind='WARRANTY')
claims = standardize_osg_columns(claims_raw, kind='CLAIMS')
store = normalize_colnames(store_raw) if store_raw is not None else pd.DataFrame()
staff_map = normalize_colnames(staff_raw) if staff_raw is not None else pd.DataFrame()

# Enrich sales with staff mapping if provided
if not staff_map.empty:
    keys = [c for c in ['store_code','staff'] if c in staff_map.columns]
    if keys:
        sales = sales.merge(staff_map.drop_duplicates(keys), on=keys, how='left')

# Filter date range
if not sales.empty:
    sales = sales[(sales['date'].dt.date >= flt_from) & (sales['date'].dt.date <= flt_to)]
if not warranty.empty:
    warranty = warranty[(warranty['date'].dt.date >= flt_from) & (warranty['date'].dt.date <= flt_to)]

# Link sales & warranty
sales_en, w_linked = merge_sales_warranty(sales, warranty)

# KPIs
kpis = compute_kpis(sales_en, w_linked)

# Group conversions
conv_dims = {
    'RBM': ['rbm'],
    'BDM': ['bdm'],
    'Staff': ['staff'],
    'Store': ['store_code'],
    'Region': ['region'],
    'Category': ['category'],
    'Brand': ['brand'],
    'Model': ['model']
}
convs = {name: group_conversion(sales_en, dims) for name, dims in conv_dims.items()}

# ML Models
prop_model, auc = (None, None)
sales_scored = pd.DataFrame()
forecast_df, forecast_model = (pd.DataFrame(), None)

if run_models and not sales_en.empty:
    prop_model, auc = build_propensity_model(sales_en)
    if prop_model is not None:
        sales_scored = score_propensity(prop_model, sales_en)

if run_models and not warranty.empty:
    forecast_df, forecast_model = build_forecast(w_linked if not w_linked.empty else warranty)

# Anomaly detection on warranty lines
anoms = detect_anomalies(w_linked if not w_linked.empty else warranty)

# Renewal pipeline: expiries in next 30/60/90 days
renewals = pd.DataFrame()
if not warranty.empty:
    w2 = warranty.copy()
    # If 'duration_year' missing, fallback to 1 year
    dur = w2['duration_year'].replace(0, np.nan).fillna(1.0)
    w2['expiry_date'] = w2['date'] + pd.to_timedelta((dur*365).round(), unit='D')
    today = pd.Timestamp.today().normalize()
    w2['days_to_expiry'] = (w2['expiry_date'] - today).dt.days
    renewals = w2[w2['days_to_expiry'].between(0, 90)].copy()
    renewals['bucket'] = pd.cut(renewals['days_to_expiry'], bins=[-1,30,60,90], labels=['0-30','31-60','61-90'])
    if show_phone_mask and 'customer_mobile' in renewals.columns:
        renewals['customer_mobile'] = renewals['customer_mobile'].apply(mask_phone)

# Mask phone numbers in UI tables
if show_phone_mask and not sales_en.empty and 'customer_mobile' in sales_en.columns:
    sales_view = sales_en.copy()
    sales_view['customer_mobile'] = sales_view['customer_mobile'].apply(mask_phone)
else:
    sales_view = sales_en

# ---------------------------
# Header
# ---------------------------
st.title("🛡️ OSG Warranty Intelligence Suite")
st.caption("Advanced analytics for warranty conversion, profitability, renewals, anomalies, and forecasting.")

# KPI Cards
kpi_cols = st.columns(6)
if kpis:
    kpi_cols[0].metric("Total Sales Lines", f"{kpis['total_sales_lines']:,}")
    kpi_cols[1].metric("Warranty Attached", f"{kpis['warranty_attached_lines']:,}")
    kpi_cols[2].metric("Conversion %", f"{kpis['conversion_rate_pct']:.2f}%")
    kpi_cols[3].metric("Warranty Revenue (Sold)", money_fmt(kpis['warranty_revenue_sold']))
    kpi_cols[4].metric("Warranty Revenue (MRP)", money_fmt(kpis['warranty_revenue_mrp']))
    kpi_cols[5].metric("Missed Revenue (Est.)", money_fmt(kpis['missed_revenue_estimate']))
else:
    st.info("Upload at least Sales and Warranty files to view KPIs.")

st.divider()

# ---------------------------
# Tabs
# ---------------------------
T1, T2, T3, T4, T5, T6, T7, T8 = st.tabs([
    "Conversion Analytics",
    "Profitability & Gaps",
    "Next-Best-Offer (ML)",
    "Forecast & Targets",
    "Renewals",
    "Anomalies",
    "Data Quality",
    "Exports"
])

# Tab 1: Conversion Analytics
with T1:
    st.subheader("Conversion by Dimensions")
    dim = st.selectbox("Group by", list(conv_dims.keys()), index=0)
    df_conv = convs.get(dim, pd.DataFrame())
    if df_conv is not None and len(df_conv)>0:
        st.dataframe(df_conv, use_container_width=True)
        # Chart
        show_top = st.slider("Top N", min_value=5, max_value=50, value=15)
        chart_df = df_conv.head(show_top)
        fig = px.bar(chart_df, x=chart_df.columns[0], y='conversion_pct', hover_data=['sales_lines','with_warranty'])
        fig.update_layout(height=420, margin=dict(l=10,r=10,b=10,t=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display. Ensure files are uploaded and columns mapped correctly.")

# Tab 2: Profitability & Gaps
with T2:
    st.subheader("Price & Discount Analysis (Warranty Lines)")
    if not w_linked.empty:
        dfp = w_linked.copy()
        dfp['discount_pct'] = np.where(dfp['plan_price']>0, (dfp['plan_price']-dfp['sold_price'])/dfp['plan_price']*100, 0)
        st.dataframe(dfp[['date','store_code','brand','model','plan_price','sold_price','discount_pct']].sort_values('discount_pct', ascending=False), use_container_width=True)
        fig = px.box(dfp, x='brand', y='sold_price', points='suspectedoutliers', title='Sold Price Distribution by Brand')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Missed Revenue Hotspots** (low conversion but high sales)")
        hot = group_conversion(sales_en, ['store_code','brand'])
        hot['rev_sold'] = sales_en.groupby(['store_code','brand'])['sold_price'].sum().reindex(hot.set_index(['store_code','brand']).index).values
        hot = hot.sort_values(['conversion_pct','rev_sold'], ascending=[True, False]).head(20)
        st.dataframe(hot, use_container_width=True)
    else:
        st.info("Upload warranty data to analyze pricing and gaps.")

# Tab 3: Next-Best-Offer (Propensity)
with T3:
    st.subheader("Who is most likely to buy warranty?")
    if prop_model is not None and not sales_scored.empty:
        st.caption(f"Model: GradientBoostingClassifier • In-sample AUC ≈ {auc:.3f}")
        topn = st.slider("Show top N prospects", 10, 500, 100)
        cols = ['invoice_number','date','customer_mobile','store_code','brand','category','model','item_rate','sold_price','rbm','bdm','staff','propensity']
        view = sales_scored[cols].sort_values('propensity', ascending=False).head(topn)
        if show_phone_mask:
            view['customer_mobile'] = view['customer_mobile'].apply(mask_phone)
        st.dataframe(view, use_container_width=True)

        fig = px.histogram(sales_scored, x='propensity', nbins=30, title='Propensity Score Distribution')
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            label='Download NBO Prospects (CSV)',
            data=sales_scored.sort_values('propensity', ascending=False)[cols].to_csv(index=False).encode('utf-8'),
            file_name='nbo_prospects.csv',
            mime='text/csv'
        )
    else:
        st.info("Need sales + warranty linked data to train propensity model.")

# Tab 4: Forecast & Targets
with T4:
    st.subheader("Warranty Sales Forecast (Store x Month)")
    if not forecast_df.empty:
        st.dataframe(forecast_df.sort_values(['store_code','ym']), use_container_width=True)
        latest = forecast_df[forecast_df['is_forecast']]
        if not latest.empty:
            fig = px.bar(latest, x='store_code', y='w_cnt', color='ym', barmode='group', title='Forecasted Warranty Count (Next 2 Months)')
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload warranty data and enable models to generate a forecast.")

# Tab 5: Renewals
with T5:
    st.subheader("Expiring in next 90 days")
    if not renewals.empty:
        cols = ['customer_mobile','invoice_number','date','expiry_date','days_to_expiry','bucket','store_code','brand','model','duration_year']
        st.dataframe(renewals[cols].sort_values(['bucket','days_to_expiry']), use_container_width=True)
        st.download_button(
            label='Download Renewal Leads (CSV)',
            data=renewals[cols].to_csv(index=False).encode('utf-8'),
            file_name='renewal_leads_next_90_days.csv',
            mime='text/csv'
        )
    else:
        st.info("No renewals in next 90 days or warranty file missing.")

# Tab 6: Anomalies
with T6:
    st.subheader("Potential Anomalies in Warranty Transactions")
    if not anoms.empty:
        bad = anoms[anoms['is_anomaly']]
        st.dataframe(bad[['date','invoice_number','store_code','brand','model','plan_price','sold_price','discount_pct']], use_container_width=True)
        fig = px.scatter(anoms, x='plan_price', y='sold_price', color='is_anomaly', hover_data=['brand','model','invoice_number'])
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomaly signals or warranty data missing.")

# Tab 7: Data Quality
with T7:
    st.subheader("Data Quality Checks")
    issues = []
    if not sales.empty:
        if sales['invoice_number'].eq('').any():
            issues.append("Sales: Blank invoice numbers present.")
        if sales['customer_mobile'].eq('').any():
            issues.append("Sales: Blank customer mobile numbers present.")
        if sales['imei'].eq('').any():
            issues.append("Sales: Blank IMEI/Serial present in some rows.")
    if not warranty.empty:
        if warranty['sold_price'].le(0).any():
            issues.append("Warranty: Some sold_price <= 0.")
        if warranty['duration_year'].eq(0).any():
            issues.append("Warranty: Missing duration_year values (defaulted to 1).")
    if not issues:
        st.success("No critical data quality issues detected.")
    else:
        for i in issues:
            st.warning(i)

    st.markdown("**Serial/IMEI Preview (sanity check)**")
    if not sales.empty:
        st.dataframe(sales[['invoice_number','imei','serial_no']].head(20), use_container_width=True)

# Tab 8: Exports
with T8:
    st.subheader("Excel Report Export")
    if kpis and (not sales_en.empty):
        data_bytes = build_excel_report(kpis, convs, sales_scored, w_linked, anoms)
        st.download_button(
            label='Download Excel Report',
            data=data_bytes,
            file_name=f'OSG_Warranty_Intelligence_{dt.date.today().isoformat()}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info("Upload and process data to enable export.")

# Footer
st.divider()
st.caption("© 2025 OSG Warranty Intelligence Suite • Built with Streamlit, scikit-learn, and Plotly")
