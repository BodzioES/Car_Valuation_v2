import psycopg2
import json
from collections import Counter
from db_config import get_db_connection


def analyze_equipment():
    """
    Identifies the 100 most frequent car features across the database and creates a feature map.
    This map is essential for transforming text equipment lists into numerical vectors for the AI model.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all equipment entries from the announcements table
    cur.execute("SELECT equipment FROM announcements")
    rows = cur.fetchall()

    all_features = []
    for row in rows:
        # Deserialize JSON string from the database into a Python list
        features = row[0] if isinstance(row[0], list) else json.loads(row[0])
        all_features.extend(features)

    # Count occurrences of every single feature
    stats = Counter(all_features)

    # Select only the top 100 most common equipment items to keep the input vector size manageable
    top_100 = stats.most_common(100)

    # Map each feature to a specific index (0 to 99)
    features_map = {feature: index for index, (feature, count) in enumerate(top_100)}

    # Save the resulting map to a JSON file for both training and GUI use
    with open("../json_files/features_map.json", "w", encoding="utf-8") as f:
        json.dump(features_map, f, ensure_ascii=False, indent=4)

    cur.close()
    conn.close()


def generate_maps():
    """
    Generates categorical mapping files for transmission, body type, and fuel.
    Each unique category value is assigned a unique integer index.
    """
    categories = ['transmission', 'body_type', 'fuel']

    conn = get_db_connection()
    cur = conn.cursor()

    for cat in categories:
        # Fetch all values for the current category
        cur.execute(f"SELECT {cat} FROM announcements")
        rows = cur.fetchall()

        # Filter out empty or null values
        all_values = [row[0] for row in rows if row[0]]

        # Get unique values present in the database
        counter = Counter(all_values)

        # Create a dictionary where each category name is a key and its index is the value
        final_map = {name: index for index, (name, count) in enumerate(counter.items())}

        # Save each category map to its respective JSON file
        file_path = f"../json_files/{cat}_map.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_map, f, ensure_ascii=False, indent=4)

        print("Successfully generated map for " + cat)

    cur.close()
    conn.close()


if __name__ == '__main__':
    # Execute the data analysis and mapping generation sequence
    analyze_equipment()
    generate_maps()