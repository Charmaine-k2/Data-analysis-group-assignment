import os
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = RAW.parent / "processed"
DB_PATH = ROOT / "db" / "south_africa.db"

LEARNING_POVERTY_CSV = RAW / "learning_poverty.csv"
PIP_CSV = RAW / "pip_poverty.csv"


def ensure_dirs():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_schema(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_poverty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            learning_poverty_rate REAL,
            UNIQUE(year)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS poverty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            poverty_rate REAL,
            gini REAL,
            UNIQUE(year)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lp_year  ON learning_poverty(year);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pov_year ON poverty(year);")
    conn.commit()


def load_learning_poverty() -> pd.DataFrame:
    df = pd.read_csv(LEARNING_POVERTY_CSV, dtype=str)
    df = df[df["REF_AREA"] == "ZAF"].copy()
    if "INDICATOR_LABEL" in df.columns:
        mask = df["INDICATOR_LABEL"].str.contains("learning poverty", case=False, na=False)
        sub = df[mask].copy()
        if sub.empty:
            sub = df.copy()
    else:
        sub = df.copy()
    sub["year"] = pd.to_numeric(sub["TIME_PERIOD"], errors="coerce")
    sub["val"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
    out = (
        sub.dropna(subset=["year", "val"])
           .groupby("year", as_index=False)["val"].mean()
           .rename(columns={"val": "learning_poverty_rate"})
    )
    return out


def load_pip_poverty() -> pd.DataFrame:
    df = pd.read_csv(PIP_CSV, dtype=str)
    df = df[df["REF_AREA"] == "ZAF"]
    df_pov = df[df["INDICATOR"].str.contains("HEADCOUNT|POVERTY", case=False, na=False)].copy()
    df_pov = df_pov.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "poverty_rate"})
    df_pov = df_pov[["year", "poverty_rate"]]
    df_gini = df[df["INDICATOR"].str.contains("GINI", na=False)].copy()
    if not df_gini.empty:
        df_gini = df_gini.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "gini"})
        df_gini = df_gini[["year", "gini"]]
        df_merge = pd.merge(df_pov, df_gini, on="year", how="left")
    else:
        df_pov["gini"] = pd.NA
        df_merge = df_pov
    df_merge["year"] = pd.to_numeric(df_merge["year"], errors="coerce")
    df_merge["poverty_rate"] = pd.to_numeric(df_merge["poverty_rate"], errors="coerce")
    df_merge["gini"] = pd.to_numeric(df_merge["gini"], errors="coerce")
    df_merge = df_merge.dropna(subset=["year"]).drop_duplicates(subset=["year"])
    return df_merge


def upsert_dataframe(conn, df: pd.DataFrame, table: str, key_cols=("year",)):
    tmp = f"tmp_{table}"
    df.to_sql(tmp, conn, if_exists="replace", index=False)
    cols = list(df.columns)
    col_list = ", ".join([f'"{c}"' for c in cols])
    key_match = " AND ".join([f'{table}."{k}" = {tmp}."{k}"' for k in key_cols])
    sql = f"""
    BEGIN;
      DELETE FROM {table}
      WHERE EXISTS (
        SELECT 1 FROM {tmp}
        WHERE {key_match}
      );
      INSERT INTO {table} ({col_list})
      SELECT {col_list} FROM {tmp};
      DROP TABLE {tmp};
    COMMIT;
    """
    conn.executescript(sql)


def safe_update_example(conn):
    with conn:
        conn.execute(
            "UPDATE learning_poverty SET learning_poverty_rate = learning_poverty_rate + ? WHERE year = ?;",
            (0.1, 2021),
        )
        conn.execute("UPDATE poverty SET poverty_rate = NULL WHERE year = ?;", (2000,))


def safe_delete_example(conn):
    with conn:
        conn.execute("DELETE FROM learning_poverty WHERE year > ?;", (2100,))
        conn.execute("DELETE FROM poverty WHERE year > ?;", (2100,))


def debug_years_and_indicators():
    lp = pd.read_csv(LEARNING_POVERTY_CSV, dtype=str)
    pip = pd.read_csv(PIP_CSV, dtype=str)
    lp_za = lp[lp["REF_AREA"] == "ZAF"].copy()
    pip_za = pip[pip["REF_AREA"] == "ZAF"].copy()
    print("\n=== LEARNING POVERTY (ZAF) ===")
    if "INDICATOR_LABEL" in lp_za.columns:
        print(lp_za.groupby("INDICATOR_LABEL")["TIME_PERIOD"].nunique().sort_values(ascending=False).head(10))
    print("Years:", sorted(lp_za["TIME_PERIOD"].dropna().unique())[:25], "...")
    print("\n=== PIP POVERTY (ZAF) ===")
    if "INDICATOR_LABEL" in pip_za.columns:
        print(pip_za.groupby("INDICATOR_LABEL")["TIME_PERIOD"].nunique().sort_values(ascending=False).head(10))
    print("Years:", sorted(pip_za["TIME_PERIOD"].dropna().unique())[:25], "...")


def get_lp_years(conn):
    q = "SELECT DISTINCT year FROM learning_poverty ORDER BY year"
    return pd.read_sql_query(q, conn)["year"].astype(int).tolist()


def build_interpolated_poverty(conn):
    df = pd.read_sql_query("SELECT year, poverty_rate, gini FROM poverty ORDER BY year", conn)
    if df.empty:
        return pd.DataFrame(columns=["year", "poverty_rate", "gini"])
    known = df.set_index("year").sort_index()
    target_years = get_lp_years(conn)
    full_index = sorted(set(known.index.tolist()) | set(target_years))
    re = known.reindex(full_index)
    re["poverty_rate"] = re["poverty_rate"].astype(float).interpolate(method="linear", limit_direction="both")
    if re["gini"].notna().sum() >= 2:
        re["gini"] = re["gini"].astype(float).interpolate(method="linear", limit_direction="both")
    else:
        re["gini"] = re["gini"].astype(float)
    interp = re.loc[target_years].reset_index().rename(columns={"index": "year"})
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS poverty_interp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                poverty_rate REAL,
                gini REAL,
                UNIQUE(year)
            );
            """
        )
    tmp = "tmp_poverty_interp"
    interp.to_sql(tmp, conn, if_exists="replace", index=False)
    with conn:
        conn.executescript(
            f"""
        DELETE FROM poverty_interp
        WHERE year IN (SELECT year FROM {tmp});
        INSERT INTO poverty_interp (year, poverty_rate, gini)
        SELECT year, poverty_rate, gini FROM {tmp};
        DROP TABLE {tmp};
        """
        )
    return interp


def join_for_analysis(conn) -> pd.DataFrame:
    inner_sql = """
    SELECT p.year,
           p.poverty_rate,
           p.gini,
           lp.learning_poverty_rate
    FROM poverty p
    INNER JOIN learning_poverty lp
      ON p.year = lp.year
    ORDER BY p.year;
    """
    df_inner = pd.read_sql_query(inner_sql, conn)
    out_inner = PROCESSED / "joined_inner.csv"
    df_inner.to_csv(out_inner, index=False)
    print(f"[INFO] INNER JOIN saved → {out_inner} (rows: {len(df_inner)})")
    full_sql = """
    SELECT p.year AS year,
           p.poverty_rate,
           p.gini,
           lp.learning_poverty_rate
    FROM poverty p
    LEFT JOIN learning_poverty lp ON p.year = lp.year
    UNION
    SELECT lp.year AS year,
           p.poverty_rate,
           p.gini,
           lp.learning_poverty_rate
    FROM learning_poverty lp
    LEFT JOIN poverty p ON lp.year = p.year
    ORDER BY year;
    """
    df_full = pd.read_sql_query(full_sql, conn)
    out_full = PROCESSED / "joined_full.csv"
    df_full.to_csv(out_full, index=False)
    print(f"[OK] FULL-OUTER style join saved → {out_full} (rows: {len(df_full)})")
    return df_full


def join_with_interpolated(conn) -> pd.DataFrame:
    sql = """
    SELECT lp.year,
           pi.poverty_rate,
           pi.gini,
           lp.learning_poverty_rate
    FROM learning_poverty lp
    JOIN poverty_interp pi ON pi.year = lp.year
    ORDER BY lp.year;
    """
    df = pd.read_sql_query(sql, conn)
    out = PROCESSED / "joined_interpolated.csv"
    df.to_csv(out, index=False)
    print(f"[OK] INTERPOLATED INNER JOIN saved → {out} (rows: {len(df)})")
    return df


def main():
    ensure_dirs()
    debug_years_and_indicators()
    if not LEARNING_POVERTY_CSV.exists():
        raise FileNotFoundError(f"Missing {LEARNING_POVERTY_CSV}.")
    if not PIP_CSV.exists():
        raise FileNotFoundError(f"Missing {PIP_CSV}.")
    conn = connect_db()
    create_schema(conn)
    lp = load_learning_poverty()
    pip = load_pip_poverty()
    upsert_dataframe(conn, lp, "learning_poverty", key_cols=("year",))
    upsert_dataframe(conn, pip, "poverty", key_cols=("year",))
    print(f"[OK] Loaded rows → learning_poverty={len(lp)}, poverty={len(pip)}")
    safe_update_example(conn)
    safe_delete_example(conn)
    print(pd.read_sql_query("SELECT * FROM learning_poverty WHERE year = 2021", conn))
    print(pd.read_sql_query("SELECT year, poverty_rate FROM poverty WHERE year = 2000", conn))
    joined = join_for_analysis(conn)
    print(joined.head())
    interp = build_interpolated_poverty(conn)
    print("[INFO] Interpolated poverty rows:", len(interp))
    joined_interp = join_with_interpolated(conn)
    print(joined_interp.head())
    conn.close()
    print(f"[OK] SQLite DB at: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
