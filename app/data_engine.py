"""
EcoSentinel Data Engine
GPU-accelerated food inspection risk analysis using RAPIDS cuDF.
Falls back to pandas if RAPIDS is not available (local dev).
"""

import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DATA_DIR", "/data")

try:
    import cudf
    import cuml
    RAPIDS = True
    print("[EcoSentinel] RAPIDS cuDF loaded — GPU acceleration active")
except ImportError:
    import pandas as cudf  # type: ignore
    RAPIDS = False
    print("[EcoSentinel] RAPIDS not found — falling back to pandas (CPU)")

import pandas as pd
import numpy as np


class EcoSentinelEngine:
    """
    Loads and joins the three SCC food inspection CSVs, computes a risk score
    per business, and exposes query methods used by both the Gradio UI and
    the OpenClaw skill API.
    """

    def __init__(self):
        self.risk_table = None
        self.raw_inspections = None
        self.raw_violations = None
        self.raw_businesses = None
        self._load_and_score()

    # ------------------------------------------------------------------
    # Internal: load, join, score
    # ------------------------------------------------------------------

    def _load_and_score(self):
        print("[EcoSentinel] Loading datasets...")

        businesses = cudf.read_csv(
            f"{DATA_DIR}/SCC_DEH_Food_Data_BUSINESS_20260306.csv",
            usecols=["business_id", "name", "address", "CITY", "STATE",
                     "postal_code", "latitude", "longitude"]
        )
        inspections = cudf.read_csv(
            f"{DATA_DIR}/SCC_DEH_Food_Data_INSPECTIONS_20260306.csv",
            usecols=["business_id", "inspection_id", "date", "SCORE",
                     "result", "description", "type", "inspection_comment"]
        )
        violations = cudf.read_csv(
            f"{DATA_DIR}/SCC_DEH_Food_Data_VIOLATIONS_20260306.csv",
            usecols=["inspection_id", "DESCRIPTION", "code", "critical",
                     "violation_comment"]
        )

        print(f"[EcoSentinel] Loaded: {len(businesses)} businesses, "
              f"{len(inspections)} inspections, {len(violations)} violations")

        # Normalise types
        inspections["SCORE"] = pd.to_numeric(
            inspections["SCORE"] if not RAPIDS else inspections["SCORE"].to_pandas(),
            errors="coerce"
        )
        inspections["date"] = pd.to_datetime(
            inspections["date"] if not RAPIDS else inspections["date"].to_pandas(),
            format="%Y%m%d", errors="coerce"
        )

        if RAPIDS:
            inspections["SCORE"] = cudf.Series(inspections["SCORE"])
            inspections["date"]  = cudf.Series(inspections["date"])

        # Store raw for history queries
        self.raw_businesses  = businesses
        self.raw_inspections = inspections
        self.raw_violations  = violations

        # --- Count critical violations per inspection ---
        if RAPIDS:
            crit_mask = violations["critical"].astype(str).str.lower() == "true"
        else:
            crit_mask = violations["critical"].astype(str).str.lower() == "true"

        critical_per_insp = (
            violations[crit_mask]
            .groupby("inspection_id")
            .size()
            .reset_index()
        )
        critical_per_insp.columns = ["inspection_id", "critical_count"]

        total_per_insp = (
            violations
            .groupby("inspection_id")
            .size()
            .reset_index()
        )
        total_per_insp.columns = ["inspection_id", "total_violation_count"]

        # Merge violation counts into inspections
        insp = inspections.merge(critical_per_insp, on="inspection_id", how="left")
        insp = insp.merge(total_per_insp, on="inspection_id", how="left")
        insp["critical_count"] = insp["critical_count"].fillna(0)
        insp["total_violation_count"] = insp["total_violation_count"].fillna(0)

        # --- Aggregate per business ---
        if RAPIDS:
            # RAPIDS groupby with cuDF
            agg_df = (
                insp.to_pandas()  # agg lambdas easier in pandas
                .groupby("business_id")
                .agg(
                    avg_score=("SCORE", "mean"),
                    min_score=("SCORE", "min"),
                    inspection_count=("inspection_id", "count"),
                    total_critical=("critical_count", "sum"),
                    total_violations=("total_violation_count", "sum"),
                    last_inspection=("date", "max"),
                    fail_count=("result", lambda x: (x.str.upper() == "F").sum()),
                )
                .reset_index()
            )
        else:
            agg_df = (
                insp
                .groupby("business_id")
                .agg(
                    avg_score=("SCORE", "mean"),
                    min_score=("SCORE", "min"),
                    inspection_count=("inspection_id", "count"),
                    total_critical=("critical_count", "sum"),
                    total_violations=("total_violation_count", "sum"),
                    last_inspection=("date", "max"),
                    fail_count=("result", lambda x: (x.str.upper() == "F").sum()),
                )
                .reset_index()
            )

        # --- Risk score (0–1, lower = more dangerous) ---
        agg_df["critical_density"] = (
            agg_df["total_critical"] / agg_df["inspection_count"].clip(lower=1)
        )
        agg_df["fail_rate"] = (
            agg_df["fail_count"] / agg_df["inspection_count"].clip(lower=1)
        )
        # Normalise avg_score to 0-1
        score_norm = (agg_df["avg_score"].fillna(0) / 100.0).clip(0, 1)
        crit_norm  = (agg_df["critical_density"] / 10.0).clip(0, 1)  # >10 crit/insp = max
        fail_norm  = agg_df["fail_rate"].clip(0, 1)

        # Higher = safer; we want higher to mean SAFER so we can sort ascending for top-risk
        agg_df["risk_score"] = (score_norm * 0.4) - (crit_norm * 0.4) - (fail_norm * 0.2)

        # --- Join business metadata ---
        biz_pd = businesses.to_pandas() if RAPIDS else businesses
        merged = agg_df.merge(
            biz_pd[["business_id", "name", "address", "CITY",
                    "postal_code", "latitude", "longitude"]],
            on="business_id",
            how="left"
        )

        # Final sort: lowest risk_score first = most dangerous
        self.risk_table = merged.sort_values("risk_score", ascending=True).reset_index(drop=True)
        print(f"[EcoSentinel] Risk table built: {len(self.risk_table)} businesses scored")

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def top_risk(self, city: str = None, n: int = 10) -> pd.DataFrame:
        """Return top N highest-risk businesses, optionally filtered by city."""
        df = self.risk_table.copy()
        if city and city.upper() != "ALL":
            df = df[df["CITY"].str.upper() == city.upper()]
        return df.head(n)[[
            "name", "CITY", "address", "avg_score", "total_critical",
            "fail_rate", "inspection_count", "last_inspection", "risk_score"
        ]].round(3)

    def business_history(self, name_query: str) -> pd.DataFrame:
        """Return inspection history for businesses matching name_query."""
        insp_pd = (
            self.raw_inspections.to_pandas() if RAPIDS else self.raw_inspections
        )
        biz_pd = (
            self.raw_businesses.to_pandas() if RAPIDS else self.raw_businesses
        )
        # Find matching business_ids
        matches = biz_pd[
            biz_pd["name"].str.upper().str.contains(name_query.upper(), na=False)
        ]
        if matches.empty:
            return pd.DataFrame()

        ids = matches["business_id"].tolist()
        history = insp_pd[insp_pd["business_id"].isin(ids)].merge(
            matches[["business_id", "name", "CITY"]], on="business_id"
        )
        return history.sort_values("date", ascending=False)

    def city_summary(self) -> pd.DataFrame:
        """Aggregate risk metrics by city for the map layer."""
        df = self.risk_table.copy()
        # Get representative lat/lon per city (mean of businesses)
        geo = df.groupby("CITY").agg(
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            avg_score=("avg_score", "mean"),
            avg_risk=("risk_score", "mean"),
            total_critical=("total_critical", "sum"),
            business_count=("business_id", "count"),
        ).reset_index().round(4)
        return geo.sort_values("avg_risk", ascending=True)  # most dangerous first

    def violation_type_summary(self) -> pd.DataFrame:
        """Most common violation types across all inspections."""
        viol_pd = (
            self.raw_violations.to_pandas() if RAPIDS else self.raw_violations
        )
        return (
            viol_pd.groupby("DESCRIPTION")
            .agg(
                count=("inspection_id", "count"),
                critical_pct=("critical", lambda x: (x.astype(str).str.lower() == "true").mean())
            )
            .reset_index()
            .sort_values("count", ascending=False)
            .head(30)
            .round(3)
        )

    def get_cities(self) -> list:
        return sorted(
            self.risk_table["CITY"].dropna().unique().tolist()
        )
