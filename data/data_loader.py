"""HDB Resale Data Loader — fetch from data.gov.sg, clean, cache locally."""

import os
import pandas as pd
import requests
import streamlit as st

API_URL = "https://data.gov.sg/api/action/datastore_search"
RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "hdb_cache.parquet")

TARGET_TOWNS = ["PUNGGOL", "SENGKANG", "HOUGANG"]


def _fetch_town(town: str) -> list:
    """Fetch all records for a single town via paginated API calls."""
    import json
    records = []
    offset = 0
    while True:
        params = {
            "resource_id": RESOURCE_ID,
            "limit": 500,
            "offset": offset,
            "filters": json.dumps({"town": town}),
            "sort": "month desc",
        }
        try:
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            batch = data["result"]["records"]
            if not batch:
                break
            records.extend(batch)
            offset += len(batch)
            if len(batch) < 500:
                break
        except Exception as e:
            st.warning(f"API error for {town} at offset {offset}: {e}")
            break
    return records


def _cache_version() -> float:
    """Return cache file modification time as version key for invalidation."""
    if os.path.exists(CACHE_FILE):
        return os.path.getmtime(CACHE_FILE)
    return 0.0


@st.cache_data(ttl=86400)
def fetch_hdb_data(force_refresh: bool = False, _version: float = None) -> pd.DataFrame:
    """Fetch HDB resale data per-town to avoid CKAN global pagination limits."""
    if _version is None:
        _version = _cache_version()

    cached = None
    if os.path.exists(CACHE_FILE):
        cached = pd.read_parquet(CACHE_FILE)
        if not force_refresh:
            return cached

    all_records = []
    try:
        for town in TARGET_TOWNS:
            st.caption(f"Fetching {town} data…")
            town_records = _fetch_town(town)
            all_records.extend(town_records)
            st.caption(f"  {town}: {len(town_records)} records")
    except Exception as e:
        st.warning(f"API unavailable ({e}), using cached data instead.")

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_parquet(CACHE_FILE, index=False)
        return df

    if cached is not None:
        return cached

    st.error("No data available. Please check network and try again.")
    return pd.DataFrame()


def clean_hdb_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare HDB resale data for analysis."""
    df = df.copy()

    # 1) town: uppercase, keep only target towns
    df["town"] = df["town"].str.upper().str.strip()
    df = df[df["town"].isin(TARGET_TOWNS)].copy()

    # 2) month → datetime
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"])
    df["year"] = df["month"].dt.year.astype(int)

    # 3) Filter 2020 onwards
    df = df[df["year"] >= 2020]

    # 4) resale_price → numeric
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")

    # 5) floor_area_sqm → numeric
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")

    # 6) lease_commence_date → compute remaining lease years
    df["lease_commence_date"] = pd.to_numeric(
        df["lease_commence_date"], errors="coerce"
    )
    current_year = pd.Timestamp.now().year
    df["remaining_lease"] = df["lease_commence_date"].apply(
        lambda x: max(99 - (current_year - x), 0) if pd.notna(x) else None
    )

    # 7) flat_type: standardise
    type_map = {
        "2 ROOM": "2-Room", "3 ROOM": "3-Room", "4 ROOM": "4-Room",
        "5 ROOM": "5-Room", "EXECUTIVE": "Executive",
        "MULTI-GENERATION": "Multi-Gen", "MULTI GENERATION": "Multi-Gen",
    }
    df["flat_type"] = df["flat_type"].str.upper().str.strip().map(type_map).fillna(df["flat_type"])

    # 8) storey_range: extract min/max/mid
    def parse_storey(s):
        try:
            parts = s.replace(" ", "").split("TO")
            low = int(parts[0])
            high = int(parts[1])
            return pd.Series([low, high, (low + high) / 2])
        except Exception:
            return pd.Series([None, None, None])

    df[["storey_low", "storey_high", "storey_mid"]] = df["storey_range"].apply(parse_storey)

    # 9) price_per_sqm
    df["price_per_sqm"] = (df["resale_price"] / df["floor_area_sqm"]).round(2)

    # 10) Drop rows with critical nulls
    critical_cols = ["resale_price", "floor_area_sqm", "flat_type", "town", "month"]
    df = df.dropna(subset=critical_cols)

    # 11) Remove outliers: price outside 3-sigma within each town+flat_type group
    for group, idx in df.groupby(["town", "flat_type"]).groups.items():
        subset = df.loc[idx, "resale_price"]
        mean, std = subset.mean(), subset.std()
        if std > 0:
            keep = (subset >= mean - 3 * std) & (subset <= mean + 3 * std)
            df = df.drop(idx[~keep])

    # 12) Remove floor_area_sqm outliers
    q_low, q_high = df["floor_area_sqm"].quantile([0.005, 0.995])
    df = df[(df["floor_area_sqm"] >= q_low) & (df["floor_area_sqm"] <= q_high)]

    df = df.reset_index(drop=True)

    st.sidebar.caption(f"Data: {len(df):,} records | {df['town'].nunique()} towns | "
                       f"{df['year'].min():.0f}-{df['year'].max():.0f}")
    return df


def load_data(force_refresh: bool = False) -> pd.DataFrame:
    """Full pipeline: fetch → clean → return."""
    version = _cache_version()
    raw = fetch_hdb_data(force_refresh, _version=version)
    return clean_hdb_data(raw)
