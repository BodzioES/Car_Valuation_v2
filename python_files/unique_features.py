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

    with open("../json_files/features_map.json", "w", encoding="utf-8") as f:
        json.dump(features_map, f, ensure_ascii=False, indent=4)

    cur.close()
    conn.close()


def generate_maps():

    categories = ['mark','transmission','body_type','fuel']

    conn = psycopg2.connect(
        host="localhost",
        database="valuation_db",
        user="postgres",
        password="lorakium1515",
    )
    cur = conn.cursor()

    for cat in categories:
        cur.execute(f"SELECT {cat} FROM announcements")
        rows = cur.fetchall()

        all_values = [row[0] for row in rows if row[0]]

        counter = Counter(all_values)

        final_map = {name: index for index, (name,count) in enumerate(counter.items())}

        file_path = f"../json_files/{cat}_map.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_map, f, ensure_ascii=False, indent=4)

        print("Successfully generated map for " + cat)

    cur.close()
    conn.close()


if __name__ == '__main__':
    analyze_equipment()
    generate_maps()
