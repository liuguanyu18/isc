from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

from .utils import infer_time_column, normalize_time_column


@dataclass
class SourceConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    table: Optional[str] = None
    query: Optional[str] = None
    limit: Optional[int] = None


def build_mysql_engine(config: SourceConfig):
    password = quote_plus(config.password or "")
    url = f"mysql+pymysql://{config.user}:{password}@{config.host}:{config.port}/{config.database}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def list_tables(engine) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SHOW TABLES"))
        return [row[0] for row in rows]


def load_data(engine, config: SourceConfig) -> pd.DataFrame:
    if config.query:
        df = pd.read_sql(text(config.query), engine)
    elif config.table:
        query = f"SELECT * FROM `{config.table}`"
        if config.limit:
            query += f" LIMIT {int(config.limit)}"
        df = pd.read_sql(query, engine)
    else:
        raise ValueError("需要提供表名或 SQL 查询")

    return df


def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    time_col = infer_time_column(df)
    if time_col:
        df = normalize_time_column(df, time_col)
    return df, time_col


def schema_summary(df: pd.DataFrame, sample_size: int = 3) -> pd.DataFrame:
    summary = []
    for col in df.columns:
        sample_values = df[col].dropna().head(sample_size).tolist()
        summary.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "sample": sample_values,
            }
        )
    return pd.DataFrame(summary)
