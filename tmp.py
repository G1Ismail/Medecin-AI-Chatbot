import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

FilesListe = [
    "src/__init__.py",
    "src/functions.py", # all the functionality
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "Logik_NoteBook/LogikNB.ipynb" # my notebook to test and create everything before creating the app
    
    #Meine App:
    "template/chatbot.html"
    "static/styles.css"
]

for filepath in FilesListe:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"creating empty file: {filepath}")
            
    else:
        logging.info(f"{filename} already exists")        