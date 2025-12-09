import requests

payload = {
    "sum_click_total": 1000,
    "n_assessments": 5,
    "eng_clicks_per_day": 30,
    "assess_per_10days": 2,
    "studied_credits": 60,
    "num_of_prev_attempts": 0,
}

r = requests.post("http://localhost:8001/predict-risk", json=payload, timeout=10)
print(r.status_code, r.json())
