from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import urllib.parse

# Database Connection Details
import os

# Database Connection Details
DB_USER = os.getenv("DB_USER", "avnadmin")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "edupath-mysql-elazzamilham2002-ac78.j.aivencloud.com")
DB_PORT = os.getenv("DB_PORT", "24128")
DB_NAME = os.getenv("DB_NAME", "edupath")

# Construct connection string with SSL parameters
# using pymysql
# ssl_ca, ssl_key etc are usually needed for strictly secure connections if not handled by system trust store.
# However, Aiven usually requires SSL. We can pass ssl={"ssl_mode": "REQUIRED"} or strictly use the URL params.
# The user provided: ?sslMode=REQUIRED&useSSL=true&requireSSL=true
# In SQLAlchemy + PyMySQL, we usually pass connect_args for SSL.

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SSL Arguments
ssl_args = {
    "ssl": {
        "ssl_mode": "REQUIRED"
    }
}

engine = create_engine(
    DATABASE_URL,
    connect_args=ssl_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
