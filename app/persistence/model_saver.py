import sqlite3 as sqlite3
from datetime import datetime

def save_result(image_name, user_id, session_id, disease_name, plant_name, label, confidence_score):
    conn = sqlite3.connect_and_create('PlantVillage.db')
    c = conn.cursor()
    c.execute('INSERT INTO prediction(orig, user_id, session_id, disease_name, plant_name, image_name, label, confidence_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
              (user_id, session_id, disease_name, plant_name, image_name, label, confidence_score, datetime.now().isoformat()))
    conn.commit()
    conn.close()
