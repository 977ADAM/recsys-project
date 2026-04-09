from __future__ import annotations

import argparse
import csv
from pathlib import Path

import psycopg
from psycopg import sql

DEFAULT_DSN = "postgresql+psycopg://recsys:recsys@127.0.0.1:5432/recsys"


def normalize_dsn(dsn: str) -> str:
    return dsn.replace("postgresql+psycopg://", "postgresql://", 1)


def read_csv_headers(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
    if not headers:
        raise ValueError(f"CSV has no header row: {csv_path}")
    return [h.strip() for h in headers]


def table_exists(cur: psycopg.Cursor, schema: str, table: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
        """,
        (schema, table),
    )
    return bool(cur.fetchone()[0])


def get_table_columns(cur: psycopg.Cursor, schema: str, table: str) -> list[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return [row[0] for row in cur.fetchall()]


def validate_columns(csv_columns: list[str], table_columns: list[str], schema: str, table: str) -> None:
    table_set = set(table_columns)
    missing_in_table = [col for col in csv_columns if col not in table_set]
    if missing_in_table:
        raise ValueError(
            f"В таблице {schema}.{table} нет колонок из CSV: {missing_in_table}. "
            f"Колонки таблицы: {table_columns}"
        )


def truncate_tables(cur: psycopg.Cursor, schema: str, users_table: str, banners_table: str) -> None:
    cur.execute(
        sql.SQL("TRUNCATE TABLE {}.{}, {}.{};").format(
            sql.Identifier(schema),
            sql.Identifier(users_table),
            sql.Identifier(schema),
            sql.Identifier(banners_table),
        )
    )


def copy_csv(cur: psycopg.Cursor, schema: str, table: str, csv_path: Path) -> None:
    csv_columns = read_csv_headers(csv_path)
    table_columns = get_table_columns(cur, schema, table)
    validate_columns(csv_columns, table_columns, schema, table)

    copy_stmt = sql.SQL(
        "COPY {}.{} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
    ).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(col) for col in csv_columns),
    )

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        with cur.copy(copy_stmt) as copy:
            while chunk := f.read(1024 * 1024):
                copy.write(chunk)



def count_rows(cur: psycopg.Cursor, schema: str, table: str) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {}.{};").format(
            sql.Identifier(schema), sql.Identifier(table)
        )
    )
    return int(cur.fetchone()[0])



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load users.csv and banners.csv into existing Postgres tables via COPY"
    )
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="Postgres DSN")
    parser.add_argument("--users-csv", default="data/db/users.csv", help="Path to users.csv")
    parser.add_argument("--banners-csv", default="data/db/banners.csv", help="Path to banners.csv")
    parser.add_argument("--schema", default="public", help="Target schema")
    parser.add_argument("--users-table", default="users", help="Existing users table name")
    parser.add_argument("--banners-table", default="banners", help="Existing banners table name")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate target tables before loading",
    )
    args = parser.parse_args()

    dsn = normalize_dsn(args.dsn)
    users_csv = Path(args.users_csv).resolve()
    banners_csv = Path(args.banners_csv).resolve()

    if not users_csv.exists():
        raise FileNotFoundError(f"users.csv not found: {users_csv}")
    if not banners_csv.exists():
        raise FileNotFoundError(f"banners.csv not found: {banners_csv}")

    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            for table_name in (args.users_table, args.banners_table):
                if not table_exists(cur, args.schema, table_name):
                    raise ValueError(f"Таблица не найдена: {args.schema}.{table_name}")

            if args.truncate:
                truncate_tables(cur, args.schema, args.users_table, args.banners_table)

            copy_csv(cur, args.schema, args.users_table, users_csv)
            copy_csv(cur, args.schema, args.banners_table, banners_csv)

            users_count = count_rows(cur, args.schema, args.users_table)
            banners_count = count_rows(cur, args.schema, args.banners_table)

        conn.commit()

    print("Загрузка завершена успешно.")
    print(f"{args.schema}.{args.users_table}: {users_count} rows")
    print(f"{args.schema}.{args.banners_table}: {banners_count} rows")


if __name__ == "__main__":
    main()
