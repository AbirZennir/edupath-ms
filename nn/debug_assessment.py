from sqlalchemy import create_engine, text
import urllib.parse
import os

# Database Connection Details
DB_USER = os.getenv("DB_USER", "avnadmin")
DB_PASS = os.getenv("DB_PASSWORD", "placeholder_password")
DB_HOST = os.getenv("DB_HOST", "edupath-mysql-elazzamilham2002-ac78.j.aivencloud.com")
DB_PORT = os.getenv("DB_PORT", "24128")
DB_NAME = "edupath"

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

def check_dates():
    with engine.connect() as conn:
        print("--- Student Assessment ---")
        count = conn.execute(text("SELECT COUNT(*) FROM student_assessment")).scalar()
        print(f"Count: {count}")
        if count > 0:
            min_date = conn.execute(text("SELECT MIN(date_submitted) FROM student_assessment")).scalar()
            max_date = conn.execute(text("SELECT MAX(date_submitted) FROM student_assessment")).scalar()
            print(f"Date Range: {min_date} to {max_date}")
        
if __name__ == "__main__":
    check_dates()
