import os
import json
import requests
from bs4 import BeautifulSoup
import time
from data_download import process_announcement

# Configuration files for persistence and vehicle makes
STATE_FILE = "../json_files/scraper_state.json"
MAKES_FILE = "../json_files/makes_dictionary.json"


def get_links_from_page(session, url):
    """
    Extracts all vehicle offer links ('/oferta/') from a specific search result page.
    """
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        links = []

        # find and collect unique offer links
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/oferta/' in href and href not in links:
                links.append(href)
        return links

    except Exception as e:
        print(f"Error: {e}")
        return []


def load_state():
    """
    Loads the last saved scraper position to resume work after a crash or restart.
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    # default starting point if no state file exists
    return {"make_index": 0, "year": 1990, "page": 1}


def save_state(make_index, year, page):
    """
    Saves the current progress (brand, year, page) to a JSON file.
    """
    with open(STATE_FILE, "w") as f:
        json.dump({"make_index": make_index, "year": year, "page": page}, f)


def start_scraping():
    """
    Main execution loop: iterates through car makes, years, and search pages.
    """
    if not os.path.exists(MAKES_FILE):
        print("Critical error: Makes dictionary missing")
        return

    # load list of car brands to iterate through
    with open(MAKES_FILE, "r", encoding="utf-8") as f:
        makes = json.load(f)

    # resume from last known state
    state = load_state()
    start_make_idx = state["make_index"]
    start_year = state["year"]
    start_page = state["page"]

    # initialize persistent HTTP session with custom User-Agent
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    # nested loops: Car Brand -> Year -> Search Result Page
    for make_idx in range(start_make_idx, len(makes)):
        make = makes[make_idx]
        first_year = start_year if make_idx == start_make_idx else 1990

        for year in range(first_year, 2025):
            first_page = start_page if (make_idx == start_make_idx and year == start_year) else 1

            for page in range(first_page, 501):
                # save progress before processing the page
                save_state(make_idx, year, page)

                # construct URL with current filters
                url = f"https://www.otomoto.pl/osobowe/{make}?search[filter_float_year%3Afrom]={year}&search[filter_float_year%3Ato]={year}&page={page}"

                current_page_links = get_links_from_page(session, url)

                # if page is empty, move to the next year
                if not current_page_links:
                    break

                # process each individual announcement link
                for link in current_page_links:
                    try:
                        process_announcement(link)
                        time.sleep(1)  # Delay to avoid being blocked
                    except Exception as e:
                        # Log errors to file for later review
                        with open("error_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"Error: {e} , {link}\n")

            # reset page counter for the next year/brand
            start_page = 1
        start_year = 1990


if __name__ == "__main__":
    start_scraping()