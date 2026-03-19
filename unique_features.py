import psycopg2
import json
from collections import Counter


def analyze_equipment():
    conn = psycopg2.connect(
        host="localhost",
        database="valuation_db",
        user="postgres",
        password="lorakium1515",
    )
    cur = conn.cursor()

    cur.execute("SELECT equipment FROM announcements")
    rows = cur.fetchall()

    all_features = []
    for row in rows:
        features = row[0] if isinstance(row[0], list) else json.loads(row[0])
        all_features.extend(features)

    stats = Counter(all_features)

    top_100 = stats.most_common(100)

    features_map = {feature: index for index, (feature, count) in enumerate(top_100)}

    with open("features_map.json", "w", encoding="utf-8") as f:
        json.dump(features_map, f, ensure_ascii=False, indent=4)

    cur.close()
    conn.close()


if __name__ == '__main__':
    analyze_equipment()
