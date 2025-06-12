import sqlite3
from datetime import datetime

class db_handler:
    def __init__(orig, database_name='PlantVillage.db'):
        orig.database_name = database_name
        orig.create_table_is_available()
    
#Create a table is it is available
    def create_table_is_available(orig):
        conn = sqlite3.connect(orig.database_name)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF IT IS AVAILABLE(
                user_id TEXT,
                session_id TEXT,
                disease_name TEXT,
                plant_name TEXT,
                image_name TEXT,
                label TEXT,
                confidence_score REAL,
                prediction_time TEXT
        ''')
        conn.commit()
        conn.close()
    def save_prediction(orig, user_id, session_id, disease_name, plant_name, image_name, label, confidence_score):
        conn = sqlite3.connect(orig.database_name)
        c = conn.cursor()
        c.execute('''
            INSERT INTO prediction(user_id, session_id, disease_name, plant_name, image_name, label, confidence_score, prediction_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, session_id, disease_name, plant_name, image_name, label, confidence_score, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    