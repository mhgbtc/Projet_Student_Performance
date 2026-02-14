import pandas as pd
import numpy as np

def load_data(filepath):
    """Charge les donnees depuis un fichier CSV"""
    return pd.read_csv(filepath)

def handle_outliers(df):
    """Traite les outliers"""
    df = df.copy()
    df.loc[df['Exam_Score'] > 100, 'Exam_Score'] = 100
    return df

def preprocess_pipeline(filepath):
    """Pipeline complet de preprocessing"""
    
    # 1. Chargement
    df = load_data(filepath)
    
    # 2. Correction des outliers
    df = handle_outliers(df)
    
    # 3. REMPLISSAGE DES VALEURS MANQUANTES
    df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
    df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
    df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])
    
    # 4. ENCODAGE ORDINAL
    ordinal_mappings = {
        'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
        'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
        'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
        'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
        'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
        'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2}
    }
    
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping)
    
    # 5. ONE-HOT ENCODING
    df = pd.get_dummies(df, columns=['School_Type'], drop_first=True)
    df = pd.get_dummies(df, columns=['Peer_Influence'], drop_first=True)
    df = pd.get_dummies(df, columns=['Extracurricular_Activities'], drop_first=True)
    df = pd.get_dummies(df, columns=['Internet_Access'], drop_first=True)
    df = pd.get_dummies(df, columns=['Learning_Disabilities'], drop_first=True)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    return df