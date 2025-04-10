import os
import random
import shutil

# Parametry
liczba_zdjec_na_klase = 1000
sciezka_bazowa = os.getcwd()
directory_name = "database_" + str(liczba_zdjec_na_klase) + "_photos"
folder_wyjsciowy = os.path.join(sciezka_bazowa, directory_name)

# Tworzy folder wyjściowy i podfoldery 0–9
os.makedirs(folder_wyjsciowy, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(folder_wyjsciowy, str(i)), exist_ok=True)

# Przetwarza każdą klasę (0–9)
for i in range(10):
    folder_klasy = os.path.join(sciezka_bazowa, str(i))
    folder_docelowy = os.path.join(folder_wyjsciowy, str(i))

    if not os.path.isdir(folder_klasy):
        print(f"⚠️ Folder '{folder_klasy}' nie istnieje, pomijam.")
        continue

    wszystkie_zdjecia = [
        f for f in os.listdir(folder_klasy)
        if os.path.isfile(os.path.join(folder_klasy, f)) and f.lower().endswith(".ppm")
    ]

    if len(wszystkie_zdjecia) == 0:
        print(f"⚠️ Brak plików .ppm w folderze '{folder_klasy}', pomijam.")
        continue

    wybrane = random.sample(wszystkie_zdjecia, min(liczba_zdjec_na_klase, len(wszystkie_zdjecia)))

    for nazwa_pliku in wybrane:
        src = os.path.join(folder_klasy, nazwa_pliku)
        dst = os.path.join(folder_docelowy, nazwa_pliku)
        shutil.copy2(src, dst)

print(f"✅ Gotowe! Losowe zdjęcia .ppm zostały skopiowane do folderu '{directory_name}'")
