from data_download import process_announcement

# Podaj link do auta, które na 100% jest bezwypadkowe w opisie
test_url = "https://www.otomoto.pl/osobowe/oferta/jaguar-xk8-ID6Huj8O.html"

print("Rozpoczynam test kolumny accident_free...")
process_announcement(test_url)
print("Test zakończony.")