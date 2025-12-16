from typing import List


def generate_recommendations(features: List[float], feature_names: List[str]) -> List[str]:
    f = dict(zip(feature_names, features))
    recs: List[str] = []

    # règles (adapte les seuils à tes stats)
    if f.get("sum_click_total", 999999) < 100:
        recs.append("Augmenter l’activité sur la plateforme (plus de clics / ressources consultées).")

    if f.get("n_assessments", 999999) < 2:
        recs.append("Soumettre davantage d’assessments et respecter les deadlines.")

    if f.get("studied_credits", 999999) < 30:
        recs.append("Augmenter le volume de crédits étudiés / suivre plus d’unités.")

    if f.get("eng_clicks_per_day", 999999) < 5:
        recs.append("Se connecter plus régulièrement (objectif : activité quotidienne).")

    if f.get("avg_score", 999999) < 50:
        recs.append("Revoir les chapitres faibles + refaire les quiz/exercices pour remonter la moyenne.")

    if not recs:
        recs.append("Comportement satisfaisant : continuer les efforts et maintenir la régularité.")

    return recs
