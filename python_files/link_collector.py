import os
import json
import requests
from bs4 import BeautifulSoup
import time
from data_download import process_announcement

STATE_FILE = "../json_files/scraper_state.json"
MAKES_FILE = "../json_files/makes_dictionary.json"

def get_links_from_page(session, url):
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/oferta/' in href and href not in links:
                links.append(href)
        return links

    except Exception as e:
        print(f"Error: {e}")
        return []

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"make_index": 0, "year": 1990, "page": 1}

def save_state(make_index, year, page):
    with open(STATE_FILE, "w") as f:
        json.dump({"make_index": make_index, "year": year, "page": page}, f)

def start_scraping():
    if not os.path.exists(MAKES_FILE):
        print("Critical error")
        return

    with open(MAKES_FILE, "r", encoding="utf-8") as f:
        makes = json.load(f)

    state = load_state()
    start_make_idx = state["make_index"]
    start_year = state["year"]
    start_page = state["page"]

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    for make_idx in range(start_make_idx, len(makes)):
        make = makes[make_idx]
        first_year = start_year if make_idx == start_make_idx else 1990

        for year in range(first_year, 2025):
            first_page = start_page if (make_idx == start_make_idx and year == start_year) else 1

            for page in range(first_page, 501):
                save_state(make_idx, year, page)
                url = f"https://www.otomoto.pl/osobowe/{make}?search[filter_float_year%3Afrom]={year}&search[filter_float_year%3Ato]={year}&page={page}"

                current_page_links = get_links_from_page(session, url)

                if not current_page_links:
                    break

                for link in current_page_links:
                    try:
                        process_announcement(link)
                        time.sleep(1)
                    except Exception as e:
                        with open("error_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"Error: {e} , {link}\n")

            start_page = 1
        start_year = 1990

if __name__ == "__main__":
    start_scraping()