"""
Tests Unitaires pour le Projet Student Performance Prediction
Ce fichier teste les fonctions de preprocessing pour s'assurer qu'elles
fonctionnent correctement et gèrent bien les cas limites.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajoute le dossier src/ au path pour importer preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import (
    load_data,
    handle_outliers,
    preprocess_pipeline
)


class TestLoadData:
    """Tests pour la fonction load_data"""
    
    def test_load_data_returns_dataframe(self):
        """Vérifie que load_data retourne bien un DataFrame"""
        # Crée un fichier CSV temporaire
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_file = 'test_temp.csv'
        test_data.to_csv(test_file, index=False)
        
        # Test
        result = load_data(test_file)
        assert isinstance(result, pd.DataFrame), "load_data doit retourner un DataFrame"
        
        # Nettoyage
        os.remove(test_file)
    
    def test_load_data_preserves_columns(self):
        """Vérifie que toutes les colonnes sont chargées"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Exam_Score': [65, 70, 75]
        })
        test_file = 'test_temp.csv'
        test_data.to_csv(test_file, index=False)
        
        result = load_data(test_file)
        assert len(result.columns) == 3, "Toutes les colonnes doivent être chargées"
        
        os.remove(test_file)
    
    def test_load_data_preserves_row_count(self):
        """Vérifie que toutes les lignes sont chargées"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20, 25, 30],
            'Exam_Score': [65, 70, 75, 80, 85]
        })
        test_file = 'test_temp.csv'
        test_data.to_csv(test_file, index=False)
        
        result = load_data(test_file)
        assert len(result) == 5, "Toutes les lignes doivent être chargées"
        
        os.remove(test_file)


class TestHandleOutliers:
    """Tests pour la fonction handle_outliers"""
    
    def test_corrects_scores_above_100(self):
        """Vérifie que les scores > 100 sont corrigés à 100"""
        df = pd.DataFrame({
            'Exam_Score': [60, 70, 101, 105, 80]
        })
        
        result = handle_outliers(df)
        
        assert result['Exam_Score'].max() == 100, "Le score max doit être 100"
        assert (result['Exam_Score'] <= 100).all(), "Aucun score ne doit dépasser 100"
    
    def test_preserves_valid_scores(self):
        """Vérifie que les scores valides ne sont pas modifiés"""
        df = pd.DataFrame({
            'Exam_Score': [55, 67, 80, 95, 100]
        })
        
        result = handle_outliers(df)
        
        assert result['Exam_Score'].tolist() == [55, 67, 80, 95, 100], \
            "Les scores valides ne doivent pas être modifiés"
    
    def test_handles_empty_dataframe(self):
        """Vérifie le comportement avec un DataFrame vide"""
        df = pd.DataFrame({'Exam_Score': []})
        
        result = handle_outliers(df)
        
        assert len(result) == 0, "Un DataFrame vide doit rester vide"
    
    def test_does_not_modify_original_dataframe(self):
        """Vérifie que le DataFrame original n'est pas modifié"""
        df = pd.DataFrame({
            'Exam_Score': [60, 70, 105]
        })
        
        original_values = df['Exam_Score'].tolist()
        result = handle_outliers(df)
        
        assert df['Exam_Score'].tolist() == original_values, \
            "Le DataFrame original ne doit pas être modifié (copie)"


class TestPreprocessPipeline:
    """Tests pour le pipeline complet de preprocessing"""
    
    def test_pipeline_removes_all_missing_values(self):
        """Vérifie que le pipeline supprime toutes les valeurs manquantes"""
        # Crée un fichier CSV avec valeurs manquantes
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Tutoring_Sessions': [0, 1, 2],
            'Family_Income': ['Low', 'Medium', 'High'],
            'Teacher_Quality': ['Low', None, 'High'],
            'School_Type': ['Public', 'Private', 'Public'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
            'Physical_Activity': [3, 4, 5],
            'Parental_Education_Level': ['College', None, 'Postgraduate'],
            'Distance_from_Home': ['Near', None, 'Far'],
            'Sleep_Hours': [7, 8, 6],
            'Previous_Scores': [60, 70, 80],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes'],
            'Internet_Access': ['Yes', 'Yes', 'No'],
            'Learning_Disabilities': ['No', 'No', 'Yes'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exam_Score': [60, 70, 101]
        })
        test_file = 'test_pipeline.csv'
        test_data.to_csv(test_file, index=False)
        
        result = preprocess_pipeline(test_file)
        
        assert result.isna().sum().sum() == 0, \
            "Le pipeline doit supprimer toutes les valeurs manquantes"
        
        os.remove(test_file)
    
    def test_pipeline_corrects_outliers(self):
        """Vérifie que le pipeline corrige les outliers"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Tutoring_Sessions': [0, 1, 2],
            'Family_Income': ['Low', 'Medium', 'High'],
            'Teacher_Quality': ['Low', 'Medium', 'High'],
            'School_Type': ['Public', 'Private', 'Public'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
            'Physical_Activity': [3, 4, 5],
            'Parental_Education_Level': ['College', 'College', 'Postgraduate'],
            'Distance_from_Home': ['Near', 'Moderate', 'Far'],
            'Sleep_Hours': [7, 8, 6],
            'Previous_Scores': [60, 70, 80],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes'],
            'Internet_Access': ['Yes', 'Yes', 'No'],
            'Learning_Disabilities': ['No', 'No', 'Yes'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exam_Score': [60, 70, 105]
        })
        test_file = 'test_pipeline.csv'
        test_data.to_csv(test_file, index=False)
        
        result = preprocess_pipeline(test_file)
        
        assert result['Exam_Score'].max() <= 100, \
            "Le pipeline doit corriger les scores > 100"
        
        os.remove(test_file)
    
    def test_pipeline_encodes_ordinal_variables(self):
        """Vérifie que le pipeline encode les variables ordinales"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Tutoring_Sessions': [0, 1, 2],
            'Family_Income': ['Low', 'Medium', 'High'],
            'Teacher_Quality': ['Low', 'Medium', 'High'],
            'School_Type': ['Public', 'Private', 'Public'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
            'Physical_Activity': [3, 4, 5],
            'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
            'Distance_from_Home': ['Near', 'Moderate', 'Far'],
            'Sleep_Hours': [7, 8, 6],
            'Previous_Scores': [60, 70, 80],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes'],
            'Internet_Access': ['Yes', 'Yes', 'No'],
            'Learning_Disabilities': ['No', 'No', 'Yes'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exam_Score': [60, 70, 80]
        })
        test_file = 'test_pipeline.csv'
        test_data.to_csv(test_file, index=False)
        
        result = preprocess_pipeline(test_file)
        
        # Vérifie que les variables ordinales sont encodées en numérique
        assert result['Motivation_Level'].dtype in [np.int64, np.float64], \
            "Motivation_Level doit être encodé en numérique"
        assert result['Motivation_Level'].tolist() == [0, 1, 2], \
            "Motivation_Level doit être mappé Low=0, Medium=1, High=2"
        
        assert result['Family_Income'].dtype in [np.int64, np.float64], \
            "Family_Income doit être encodé en numérique"
        
        os.remove(test_file)
    
    def test_pipeline_creates_onehot_columns(self):
        """Vérifie que le pipeline crée des colonnes one-hot"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Tutoring_Sessions': [0, 1, 2],
            'Family_Income': ['Low', 'Medium', 'High'],
            'Teacher_Quality': ['Low', 'Medium', 'High'],
            'School_Type': ['Public', 'Private', 'Public'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
            'Physical_Activity': [3, 4, 5],
            'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
            'Distance_from_Home': ['Near', 'Moderate', 'Far'],
            'Sleep_Hours': [7, 8, 6],
            'Previous_Scores': [60, 70, 80],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes'],
            'Internet_Access': ['Yes', 'Yes', 'No'],
            'Learning_Disabilities': ['No', 'No', 'Yes'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exam_Score': [60, 70, 80]
        })
        test_file = 'test_pipeline.csv'
        test_data.to_csv(test_file, index=False)
        
        result = preprocess_pipeline(test_file)
        
        # Vérifie que les colonnes one-hot ont été créées
        # Gender == Gender_Male (drop_first=True)
        # School_Type == School_Type_Public ou School_Type_Private
        
        assert any('Gender' in col for col in result.columns), \
            "Une colonne Gender encodée doit exister"
        assert any('School_Type' in col for col in result.columns), \
            "Une colonne School_Type encodée doit exister"
        
        # Vérifie que les colonnes originales n'existent plus
        assert 'Gender' not in result.columns or result['Gender'].dtype in [np.int64, bool], \
            "Gender doit être encodée"
        
        os.remove(test_file)
    
    def test_pipeline_preserves_row_count(self):
        """Vérifie que le pipeline ne perd pas de lignes"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20, 25],
            'Attendance': [80, 90, 95, 85],
            'Parental_Involvement': ['Low', 'Medium', 'High', 'Low'],
            'Access_to_Resources': ['Low', 'Medium', 'High', 'Medium'],
            'Motivation_Level': ['Low', 'Medium', 'High', 'Low'],
            'Tutoring_Sessions': [0, 1, 2, 1],
            'Family_Income': ['Low', 'Medium', 'High', 'Low'],
            'Teacher_Quality': ['Low', 'Medium', None, 'High'],
            'School_Type': ['Public', 'Private', 'Public', 'Private'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative', 'Positive'],
            'Physical_Activity': [3, 4, 5, 3],
            'Parental_Education_Level': ['College', None, 'Postgraduate', 'High School'],
            'Distance_from_Home': ['Near', 'Moderate', None, 'Far'],
            'Sleep_Hours': [7, 8, 6, 7],
            'Previous_Scores': [60, 70, 80, 65],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes', 'No'],
            'Internet_Access': ['Yes', 'Yes', 'No', 'Yes'],
            'Learning_Disabilities': ['No', 'No', 'Yes', 'No'],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Exam_Score': [60, 70, 101, 75]
        })
        test_file = 'test_pipeline.csv'
        test_data.to_csv(test_file, index=False)
        
        result = preprocess_pipeline(test_file)
        
        assert len(result) == len(test_data), \
            "Le pipeline ne doit pas perdre de lignes"
        
        os.remove(test_file)
    
    def test_pipeline_preserves_numeric_columns(self):
        """Vérifie que les colonnes numériques sont préservées"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Tutoring_Sessions': [0, 1, 2],
            'Family_Income': ['Low', 'Medium', 'High'],
            'Teacher_Quality': ['Low', 'Medium', 'High'],
            'School_Type': ['Public', 'Private', 'Public'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
            'Physical_Activity': [3, 4, 5],
            'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
            'Distance_from_Home': ['Near', 'Moderate', 'Far'],
            'Sleep_Hours': [7, 8, 6],
            'Previous_Scores': [60, 70, 80],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes'],
            'Internet_Access': ['Yes', 'Yes', 'No'],
            'Learning_Disabilities': ['No', 'No', 'Yes'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exam_Score': [65, 70, 75]
        })
        test_file = 'test_pipeline.csv'
        test_data.to_csv(test_file, index=False)
        
        result = preprocess_pipeline(test_file)
        
        # Vérifie que les colonnes numériques existent toujours
        assert 'Hours_Studied' in result.columns, "Hours_Studied doit être préservé"
        assert 'Attendance' in result.columns, "Attendance doit être préservé"
        assert 'Exam_Score' in result.columns, "Exam_Score doit être préservé"
        
        # Vérifie que les valeurs n'ont pas changé
        assert result['Hours_Studied'].tolist() == [10, 15, 20], \
            "Hours_Studied ne doit pas être modifié"
        assert result['Exam_Score'].tolist() == [65, 70, 75], \
            "Exam_Score ne doit pas être modifié"
        
        os.remove(test_file)


class TestDataIntegrity:
    """Tests d'intégrité des données après preprocessing"""
    
    def test_reproducibility(self):
        """Vérifie que le preprocessing est reproductible"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Tutoring_Sessions': [0, 1, 2],
            'Family_Income': ['Low', 'Medium', 'High'],
            'Teacher_Quality': ['Low', None, 'High'],
            'School_Type': ['Public', 'Private', 'Public'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
            'Physical_Activity': [3, 4, 5],
            'Parental_Education_Level': ['High School', None, 'Postgraduate'],
            'Distance_from_Home': ['Near', None, 'Far'],
            'Sleep_Hours': [7, 8, 6],
            'Previous_Scores': [60, 70, 80],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes'],
            'Internet_Access': ['Yes', 'Yes', 'No'],
            'Learning_Disabilities': ['No', 'No', 'Yes'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exam_Score': [60, 70, 80]
        })
        test_file = 'test_repro.csv'
        test_data.to_csv(test_file, index=False)
        
        # Exécute le pipeline deux fois
        result1 = preprocess_pipeline(test_file)
        result2 = preprocess_pipeline(test_file)
        
        # Les résultats doivent être identiques
        pd.testing.assert_frame_equal(result1, result2, 
            "Le preprocessing doit être reproductible")
        
        os.remove(test_file)
    
    def test_exam_score_not_leaked(self):
        """Vérifie que Exam_Score n'est pas modifié (sauf outliers)"""
        test_data = pd.DataFrame({
            'Hours_Studied': [10, 15, 20],
            'Attendance': [80, 90, 95],
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Tutoring_Sessions': [0, 1, 2],
            'Family_Income': ['Low', 'Medium', 'High'],
            'Teacher_Quality': ['Low', 'Medium', 'High'],
            'School_Type': ['Public', 'Private', 'Public'],
            'Peer_Influence': ['Positive', 'Neutral', 'Negative'],
            'Physical_Activity': [3, 4, 5],
            'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
            'Distance_from_Home': ['Near', 'Moderate', 'Far'],
            'Sleep_Hours': [7, 8, 6],
            'Previous_Scores': [60, 70, 80],
            'Extracurricular_Activities': ['Yes', 'No', 'Yes'],
            'Internet_Access': ['Yes', 'Yes', 'No'],
            'Learning_Disabilities': ['No', 'No', 'Yes'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exam_Score': [60, 70, 80]
        })
        test_file = 'test_leak.csv'
        test_data.to_csv(test_file, index=False)
        
        result = preprocess_pipeline(test_file)
        
        # Vérifie que Exam_Score existe et contient les bonnes valeurs
        assert 'Exam_Score' in result.columns, "Exam_Score doit être présent"
        assert result['Exam_Score'].tolist() == [60, 70, 80], \
            "Exam_Score ne doit pas être modifié (pas d'outliers ici)"
        
        os.remove(test_file)


# Point d'entrée pour exécuter les tests
if __name__ == "__main__":
    
    # Exécuter pytest avec verbose
    exit_code = pytest.main([__file__, "-v", "--tb=short", "-W", "ignore::DeprecationWarning"])
    
    if exit_code == 0:
        print("TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS")
    else:
        print(f"CERTAINS TESTS ONT ÉCHOUÉ (Code de sortie: {exit_code})")