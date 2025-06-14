import sqlite3
from datetime import datetime

class db_handler:
    def __init__(orig, database_name='PlantVillage.db'):
        orig.database_name = database_name
        orig.create_table_if_not_exists()
    
#Create a table if it is available
    def create_table_if_not_exists(orig):
        conn = sqlite3.connect(orig.database_name)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS(
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
    # Save the results of new prediction to the database
    def save_prediction(orig, user_id, session_id, disease_name, plant_name, image_name, label, confidence_score):
        conn = sqlite3.connect(orig.database_name)
        c = conn.cursor()
        # ? is used for secure data injection method to prevent SQL injection
        c.execute('''
            INSERT INTO prediction(user_id, session_id, disease_name, plant_name, image_name, label, confidence_score, prediction_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, session_id, disease_name, plant_name, image_name, label, confidence_score, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    