import requests
from bs4 import BeautifulSoup
import json
import psycopg2

from db_config import get_db_connection

# Database connection credentials
conn = get_db_connection()


def process_announcement(url_announcement):
    """
    Downloads, parses, and saves vehicle data from a single announcement URL to the database.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # Execute HTTP GET request to fetch the page content
    request = requests.get(url_announcement, headers=headers)

    if request.status_code != 200:
        print(f"Connection error: {request.status_code}")
        return

    soup = BeautifulSoup(request.text, 'html.parser')

    # Locate the __NEXT_DATA__ script tag containing the structured JSON data
    script_json = soup.find('script', id='__NEXT_DATA__')

    if not script_json:
        print("JSON block not found")
        return

    # Parse raw string content into a python dictionary
    data_raw = json.loads(script_json.string)

    try:
        # Navigate to the core advertisement and parameters section
        car_data = data_raw.get('props', {}).get('pageProps', {}).get('advert', {})
        parameters = car_data.get('parametersDict', {})
        id_announcement = str(car_data.get('id', ''))

        def extract_data(field):
            """
            Helper function to safely extract labels or values from complex JSON structures.
            """
            if isinstance(field, dict):
                values = field.get('values', [])
                if isinstance(values, list) and len(values) > 0:
                    first_val = values[0]
                    if isinstance(first_val, dict):
                        return str(first_val.get('label', first_val.get('value', 'Lack')))
                return str(field.get('value', 'Lack'))
            return str(field) if field else 'Lack'

        # Mapping basic vehicle information
        mark = extract_data(parameters.get('make'))
        model = extract_data(parameters.get('model'))
        fuel = extract_data(parameters.get('fuel_type', parameters.get('fuel')))
        body_type = extract_data(parameters.get('body_type'))
        transmission = extract_data(parameters.get('gearbox', parameters.get('transmission')))

        # Numeric data cleaning: Year of production
        year_raw = extract_data(parameters.get('year')).replace(' ', '')
        year_production = int(year_raw) if year_raw.isdigit() else 0

        # Numeric data cleaning: Mileage (course)
        course_raw = extract_data(parameters.get('mileage')).replace(' ', '').replace('km', '')
        course = int(course_raw) if course_raw.isdigit() else 0

        # Numeric data cleaning: Engine capacity (handling spaces, units, and decimal commas)
        capacity_raw = extract_data(parameters.get('engine_capacity')).replace(' ', '').replace('cm3', '').replace(',',
                                                                                                                   '.')
        capacity = float(capacity_raw) if capacity_raw.replace('.', '').isdigit() else 0.0

        # Numeric data cleaning: Engine power
        power_raw = extract_data(parameters.get('engine_power')).replace(' ', '').replace('KM', '')
        power = int(power_raw) if power_raw.isdigit() else 0

        # Collision-free logic: determining status based on 'no_accident' and 'damaged' flags
        is_damaged = extract_data(parameters.get('damaged')).strip().capitalize()
        no_accident = extract_data(parameters.get('no_accident')).strip().capitalize()

        if no_accident == 'Tak':
            accident_free = True
        elif is_damaged == 'Tak':
            accident_free = False
        else:
            accident_free = True if is_damaged == 'Nie' else False

        # Equipment extraction: flattening nested categories into a single list
        equipment_raw = car_data.get('equipment', [])
        labels_lists = []

        for category in equipment_raw:
            category_values = category.get('values', [])
            for item in category_values:
                label = item.get('label', '')
                if label:
                    labels_lists.append(label)

        # Serialize equipment list to a JSON string for database storage
        equipment = json.dumps(labels_lists, ensure_ascii=False)

        # Price extraction and currency verification
        price_data = car_data.get('price', {})
        price = 0
        currency = 'PLN'

        if isinstance(price_data, dict):
            price = int(price_data.get('amount', price_data.get('value', 0)))
            currency = price_data.get('currency', 'PLN')

        # Data validation: Ensure record has ID, valid price, and is in PLN
        if not id_announcement or price == 0 or currency != 'PLN':
            print("Data incomplete")
            return

        # Database connection and execution
        conn = get_db_connection()
        cursor = conn.cursor()

        # SQL query: Inserts new record or updates existing one if ID conflict occurs
        query_sql = """
                    INSERT INTO announcements (id_announcement, mark, model, year_production, course, power_hp,
                                               capacity_cm3, fuel, transmission, body_type, accident_free, price, \
                                               equipment)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id_announcement) DO
                    UPDATE SET
                        price = EXCLUDED.price,
                        course = EXCLUDED.course,
                        accident_free = EXCLUDED.accident_free,
                        equipment = EXCLUDED.equipment;
                    """

        data_to_save = (id_announcement, mark, model, year_production, course, power,
                        capacity, fuel, transmission, body_type, accident_free, price, equipment)

        cursor.execute(query_sql, data_to_save)

        conn.commit()
        cursor.close()
        conn.close()

        print(f"Saved: {mark} {model}")

    except Exception as e:
        print(f"Error: {e}")