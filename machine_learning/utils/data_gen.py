import random
from datetime import datetime, timedelta, timezone

def get_current_utc_time() -> str:
    """Returns current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def get_random_date(days_back: int = 30) -> str:
    """Returns a random date within the last N days in YYYY-MM-DD format."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    random_days = random.randrange(days_back)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime('%Y-%m-%d')

def generate_risk_score() -> float:
    """Generates a random risk score between 0 and 1."""
    return round(random.uniform(0, 1), 2)
