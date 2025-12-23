from sqlalchemy import create_engine, inspect, text
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

def inspect_student_vle():
    inspector = inspect(engine)
    print("Columns in student_vle:", 
          [c['name'] for c in inspector.get_columns("student_vle")])
    
    with engine.connect() as conn:
        # Count rows
        count = conn.execute(text("SELECT COUNT(*) FROM student_vle")).scalar()
        print(f"Total rows in student_vle: {count}")
        
        # Check date range
        min_date = conn.execute(text("SELECT MIN(date) FROM student_vle")).scalar()
        max_date = conn.execute(text("SELECT MAX(date) FROM student_vle")).scalar()
        print(f"Date range: {min_date} to {max_date}")

        # Sample some positive dates
        sample = conn.execute(text("SELECT date, sum_click FROM student_vle WHERE date >= 0 LIMIT 5")).fetchall()
        print("Sample data (date >= 0):", sample)

if __name__ == "__main__":
    inspect_student_vle()
