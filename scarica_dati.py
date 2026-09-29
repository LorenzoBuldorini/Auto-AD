"""
Scarica l'intera cartella ABU-Airport da Google Drive in data/ABU_Airport/.
"""
import os
import gdown

FOLDER_ID = '1h8_l-elrh5thJe7bELAMMt6ATZ8hFgvM'

def scarica_dataset(cartella_destinazione='data/ABU_Airport'):
    os.makedirs(cartella_destinazione, exist_ok=True)
    gdown.download_folder(id=FOLDER_ID, output=cartella_destinazione,
                           quiet=False, use_cookies=False)

if __name__ == "__main__":
    scarica_dataset()
