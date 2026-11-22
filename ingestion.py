import os
import json
import glob
import pandas as pd


def load_aavail_data(data_dir: str) -> pd.DataFrame:
    """Load and combine daily JSON files from a directory into a DataFrame."""
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    records = []

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)

    if not records:
        return pd.DataFrame(columns=["date", "country", "revenue", "purchases", "total_views"])

    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def summarize_daily_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric metrics by date and country."""
    if df.empty:
        return df

    agg_map = {}
    for col in ["revenue", "purchases", "total_views"]:
        if col in df.columns:
            agg_map[col] = "sum"

    if "country" in df.columns and agg_map:
        return df.groupby(["date", "country"], as_index=False).agg(agg_map)
    return df


if __name__ == "__main__":
    sample_df = load_aavail_data("cs-train")
    print(f"Loaded rows: {len(sample_df)}")
