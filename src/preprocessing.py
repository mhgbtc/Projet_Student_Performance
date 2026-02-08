import pandas as pd

def load_data(filepath):
    """Charge les données"""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Gère les valeurs manquantes"""
    df = df.copy()
    
    # Imputation simple par le mode pour les catégorielles
    for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def handle_outliers(df):
    """Traite les outliers"""
    df = df.copy()
    
    # Cap Exam_Score à 100
    df.loc[df['Exam_Score'] > 100, 'Exam_Score'] = 100
    
    return df

def encode_categorical(df):
    """Encode les variables catégorielles"""
    df = df.copy()
    
    # Encodage ordinal
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
    
    # One-hot encoding pour les autres
    df = pd.get_dummies(df, columns=['School_Type', 'Peer_Influence', 
                                     'Extracurricular_Activities', 'Internet_Access',
                                     'Learning_Disabilities', 'Gender'], 
                        drop_first=True)
    
    return df

def preprocess_pipeline(filepath):
    """Pipeline complet de preprocessing"""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_categorical(df)
    
    return df