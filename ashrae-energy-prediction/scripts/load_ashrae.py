"""ASHRAE Great Energy Predictor III 데이터를 Kaggle에서 받아 PostgreSQL에 적재.

스키마 (ashrae)
  site               (site_id PK)
  meter_type         (meter PK, name)                      0=electricity 1=chilledwater 2=steam 3=hotwater
  building_metadata  (building_id PK, site_id FK→site)
  weather_train/test (PK site_id+year+month+day+hour, site_id FK→site)
  test               (row_id PK, building_id FK→building_metadata, meter FK→meter_type,
                      UNIQUE building_id+meter+year+month+day+hour)
  train              (PK building_id+meter+year+month+day+hour, 동일 FK)   --include-train 시에만

timestamp는 원본 그대로 저장하지 않고 year / month / day / hour / day_of_week(ISO, 1=월~7=일)로 분리한다.
CSV → UNLOGGED 스테이징 테이블(COPY) → INSERT ... SELECT EXTRACT(...) → 제약조건/인덱스 생성 순으로 처리.

사전 조건
  - Kaggle 대회 규칙 동의, 토큰: KAGGLE_API_TOKEN 환경변수 또는 ../kaggleToken 파일
  - PostgreSQL 접속: PGHOST, PGPORT, PGUSER, PGPASSWORD (libpq 환경변수)

사용
  python scripts/load_ashrae.py                   # train 제외 전부 (building, weather_train/test, test)
  python scripts/load_ashrae.py --include-train   # train까지
"""

import argparse
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psycopg
from psycopg import sql

COMPETITION = "ashrae-energy-prediction"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ashrae"
TOKEN_FILE = ROOT.parent / "kaggleToken"

METER_TYPES = [(0, "electricity"), (1, "chilledwater"), (2, "steam"), (3, "hotwater")]

# 날짜 분리 컬럼 (DDL / SELECT 식)
TIME_COLS_DDL = """
    year        SMALLINT NOT NULL,
    month       SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    day         SMALLINT NOT NULL CHECK (day BETWEEN 1 AND 31),
    hour        SMALLINT NOT NULL CHECK (hour BETWEEN 0 AND 23),
    day_of_week SMALLINT NOT NULL CHECK (day_of_week BETWEEN 1 AND 7)"""
TIME_COLS_SELECT = """
    EXTRACT(YEAR FROM ts)::smallint, EXTRACT(MONTH FROM ts)::smallint,
    EXTRACT(DAY FROM ts)::smallint,  EXTRACT(HOUR FROM ts)::smallint,
    EXTRACT(ISODOW FROM ts)::smallint"""
TIME_KEY = "year, month, day, hour"

WEATHER_COLS = ("air_temperature, cloud_coverage, dew_temperature, precip_depth_1_hr, "
                "sea_level_pressure, wind_direction, wind_speed")
WEATHER_STAGE = """
    site_id SMALLINT, ts TIMESTAMP,
    air_temperature REAL, cloud_coverage REAL, dew_temperature REAL, precip_depth_1_hr REAL,
    sea_level_pressure REAL, wind_direction REAL, wind_speed REAL"""
WEATHER_DDL = f"""
    site_id SMALLINT NOT NULL,{TIME_COLS_DDL},
    air_temperature REAL, cloud_coverage REAL, dew_temperature REAL, precip_depth_1_hr REAL,
    sea_level_pressure REAL, wind_direction REAL, wind_speed REAL"""


def weather_spec(name: str) -> dict:
    return {
        "csv": f"{name}.csv",
        "stage": WEATHER_STAGE,
        "ddl": WEATHER_DDL,
        "insert": f"site_id, {TIME_KEY}, day_of_week, {WEATHER_COLS}",
        "select": f"site_id, {TIME_COLS_SELECT}, {WEATHER_COLS}",
        "constraints": [
            f"ADD CONSTRAINT pk_{name} PRIMARY KEY (site_id, {TIME_KEY})",
            f"ADD CONSTRAINT fk_{name}_site FOREIGN KEY (site_id) REFERENCES {{s}}.site (site_id)",
        ],
        "indexes": [f"CREATE INDEX ix_{name}_date ON {{s}}.{name} (year, month, day)"],
    }


# 적재 순서 = FK 의존 순서
TABLES: dict[str, dict] = {
    "building_metadata": {
        "csv": "building_metadata.csv",
        "stage": """site_id SMALLINT, building_id INTEGER, primary_use TEXT, square_feet INTEGER,
                    year_built SMALLINT, floor_count SMALLINT""",
        "ddl": """site_id SMALLINT NOT NULL, building_id INTEGER NOT NULL, primary_use TEXT NOT NULL,
                  square_feet INTEGER NOT NULL CHECK (square_feet > 0),
                  year_built SMALLINT, floor_count SMALLINT""",
        "insert": "site_id, building_id, primary_use, square_feet, year_built, floor_count",
        "select": "site_id, building_id, primary_use, square_feet, year_built, floor_count",
        "constraints": [
            "ADD CONSTRAINT pk_building_metadata PRIMARY KEY (building_id)",
            "ADD CONSTRAINT fk_building_site FOREIGN KEY (site_id) REFERENCES {s}.site (site_id)",
        ],
        "indexes": [
            "CREATE INDEX ix_building_site ON {s}.building_metadata (site_id)",
            "CREATE INDEX ix_building_primary_use ON {s}.building_metadata (primary_use)",
        ],
    },
    "weather_train": weather_spec("weather_train"),
    "weather_test": weather_spec("weather_test"),
    "test": {
        "csv": "test.csv",
        "stage": "row_id INTEGER, building_id INTEGER, meter SMALLINT, ts TIMESTAMP",
        "ddl": f"row_id INTEGER NOT NULL, building_id INTEGER NOT NULL, meter SMALLINT NOT NULL,{TIME_COLS_DDL}",
        "insert": f"row_id, building_id, meter, {TIME_KEY}, day_of_week",
        "select": f"row_id, building_id, meter, {TIME_COLS_SELECT}",
        "constraints": [
            "ADD CONSTRAINT pk_test PRIMARY KEY (row_id)",
            f"ADD CONSTRAINT uq_test_bld_meter_time UNIQUE (building_id, meter, {TIME_KEY})",
            "ADD CONSTRAINT fk_test_building FOREIGN KEY (building_id) "
            "REFERENCES {s}.building_metadata (building_id)",
            "ADD CONSTRAINT fk_test_meter FOREIGN KEY (meter) REFERENCES {s}.meter_type (meter)",
        ],
        "indexes": [
            "CREATE INDEX ix_test_date ON {s}.test (year, month, day)",
            "CREATE INDEX ix_test_meter ON {s}.test (meter)",
        ],
    },
    "train": {
        "csv": "train.csv",
        "stage": "building_id INTEGER, meter SMALLINT, ts TIMESTAMP, meter_reading DOUBLE PRECISION",
        "ddl": f"building_id INTEGER NOT NULL, meter SMALLINT NOT NULL,{TIME_COLS_DDL},"
               "\n    meter_reading DOUBLE PRECISION NOT NULL",
        "insert": f"building_id, meter, {TIME_KEY}, day_of_week, meter_reading",
        "select": f"building_id, meter, {TIME_COLS_SELECT}, meter_reading",
        "constraints": [
            f"ADD CONSTRAINT pk_train PRIMARY KEY (building_id, meter, {TIME_KEY})",
            "ADD CONSTRAINT fk_train_building FOREIGN KEY (building_id) "
            "REFERENCES {s}.building_metadata (building_id)",
            "ADD CONSTRAINT fk_train_meter FOREIGN KEY (meter) REFERENCES {s}.meter_type (meter)",
        ],
        "indexes": [
            "CREATE INDEX ix_train_date ON {s}.train (year, month, day)",
            "CREATE INDEX ix_train_meter ON {s}.train (meter)",
        ],
    },
}


def set_data_dir(path: Path) -> None:
    global DATA_DIR
    DATA_DIR = path


def kaggle_env() -> dict:
    env = dict(os.environ)
    if "KAGGLE_API_TOKEN" not in env and TOKEN_FILE.exists():
        env["KAGGLE_API_TOKEN"] = TOKEN_FILE.read_text(encoding="utf-8").strip()
    return env


def download(tables: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for t in tables:
        csv = DATA_DIR / TABLES[t]["csv"]
        if csv.exists():
            continue
        print(f"Downloading {csv.name}")
        subprocess.run(
            [sys.executable, "-c", "from kaggle.cli import main; main()",
             "competitions", "download", "-c", COMPETITION, "-f", csv.name, "-p", str(DATA_DIR), "-q"],
            check=True, env=kaggle_env(),
        )
        for z in DATA_DIR.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(DATA_DIR)
            z.unlink()


def ensure_database(dbname: str) -> None:
    with psycopg.connect(dbname="postgres", autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
            print(f"Created database {dbname}")


def q(conn: psycopg.Connection, stmt: str, s: str):
    return conn.execute(stmt.replace("{s}", s))


def create_lookups(conn: psycopg.Connection, s: str) -> None:
    q(conn, "CREATE TABLE {s}.meter_type (meter SMALLINT PRIMARY KEY, name TEXT NOT NULL UNIQUE)", s)
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {s}.meter_type VALUES (%s, %s)", METER_TYPES)
    # site 는 원본 파일이 없으므로 building / weather CSV 의 site_id 로 채운다 (load_table 에서 upsert)
    q(conn, "CREATE TABLE {s}.site (site_id SMALLINT PRIMARY KEY)", s)


def load_table(conn: psycopg.Connection, s: str, table: str) -> None:
    spec = TABLES[table]
    t0 = time.time()
    stage = f"stg_{table}"

    q(conn, f"CREATE UNLOGGED TABLE {{s}}.{stage} ({spec['stage']})", s)
    copy_sql = f"COPY {s}.{stage} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
    with conn.cursor() as cur, cur.copy(copy_sql) as copy, open(DATA_DIR / spec["csv"], "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            copy.write(chunk)

    if "site_id" in spec["stage"]:
        q(conn, f"INSERT INTO {{s}}.site SELECT DISTINCT site_id FROM {{s}}.{stage} "
                "ON CONFLICT DO NOTHING", s)

    q(conn, f"CREATE TABLE {{s}}.{table} ({spec['ddl']})", s)
    rows = q(conn, f"INSERT INTO {{s}}.{table} ({spec['insert']}) "
                   f"SELECT {spec['select']} FROM {{s}}.{stage}", s).rowcount
    q(conn, f"DROP TABLE {{s}}.{stage}", s)
    t1 = time.time()

    # 대량 적재 후 제약조건/인덱스 생성 (적재 중에 두는 것보다 훨씬 빠름)
    for c in spec["constraints"]:
        q(conn, f"ALTER TABLE {{s}}.{table} {c}", s)
    for ix in spec["indexes"]:
        q(conn, ix, s)
    conn.commit()
    print(f"  {table:<18} {rows:>12,} rows  load {t1 - t0:6.1f}s  keys/index {time.time() - t1:6.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("PGDATABASE", "ashrae"))
    ap.add_argument("--schema", default="ashrae")
    ap.add_argument("--include-train", action="store_true", help="train.csv 까지 적재")
    ap.add_argument("--only", nargs="+", choices=list(TABLES),
                    help="지정한 테이블만 (재)적재. 스키마/다른 테이블은 유지")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    set_data_dir(args.data_dir)

    if args.only:
        tables = [t for t in TABLES if t in args.only]
    else:
        tables = [t for t in TABLES if args.include_train or t != "train"]
    if not args.skip_download:
        download(tables)
    ensure_database(args.db)

    s = args.schema
    with psycopg.connect(dbname=args.db) as conn:
        if args.only:
            # 부분 적재: 대상 테이블만 지우고 다시 넣는다 (FK 참조 테이블은 이미 있어야 함)
            for t in tables:
                conn.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(s, t)))
                conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(s, f"stg_{t}")))
        else:
            # 전체 적재: 스키마를 통째로 다시 만든다
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(s)))
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(s)))
            create_lookups(conn, s)
        conn.commit()
        print(f"Loading into {args.db}.{s}: {', '.join(tables)}")
        for t in tables:
            load_table(conn, s, t)
        conn.execute("ANALYZE")
    print("Done")


if __name__ == "__main__":
    main()
