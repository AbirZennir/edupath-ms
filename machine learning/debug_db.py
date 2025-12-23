from sqlalchemy import create_engine, inspect
import urllib.parse
import os

# Database Connection Details
DB_USER = os.getenv("DB_USER", "avnadmin")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "edupath-mysql-elazzamilham2002-ac78.j.aivencloud.com")
DB_PORT = os.getenv("DB_PORT", "24128")
DB_NAME = os.getenv("DB_NAME", "edupath")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SSL Arguments
ssl_args = {
    "ssl": {
        "ssl_mode": "REQUIRED"
    }
}

engine = create_engine(
    DATABASE_URL,
    connect_args=ssl_args
)

def list_tables():
    inspector = inspect(engine)
    print("Tables in DB:", inspector.get_table_names())

if __name__ == "__main__":
    list_tables()
