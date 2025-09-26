"""
SSEL-CPP: Soft-voting ensemble model for Cell Penetrating Peptide prediction
Based on XGBoost + LightGBM ensemble with different feature sets
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from .utils.fasta_parser import FastaParser
from .feature_extraction import MordredExtractor

class SSELCPP:
    def __init__(self, model_dir="models"):
        """Initialize SSEL-CPP with pre-trained models and feature sets"""
        self.model_dir = Path(model_dir)
        self.fasta_parser = FastaParser()
        self.feature_extractor = MordredExtractor(model_dir)
        
        # Load pre-trained models and their feature sets
        self._load_models()
        self._load_features()
        
    def _load_models(self):
        """Load XGBoost and LightGBM models"""
        try:
            self.xgb_model = joblib.load(self.model_dir / "xgb_model.pkl")
            self.lgb_model = joblib.load(self.model_dir / "lgb_model.pkl")
            print("Models loaded successfully!")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found in {self.model_dir}. {e}")
    
    def _load_features(self):
        """Load feature lists for each model (subsets of 986 features)"""
        try:
            # Load XGBoost features (subset of 986)
            xgb_features_df = pd.read_csv(self.model_dir / "xgb_features.csv")
            self.xgb_features = xgb_features_df["Feature"].tolist()
            
            # Load LightGBM features (subset of 986)
            lgb_features_df = pd.read_csv(self.model_dir / "lgb_features.csv") 
            self.lgb_features = lgb_features_df["Feature"].tolist()
            
            print(f"Features loaded: XGB={len(self.xgb_features)}, LGB={len(self.lgb_features)} (from 986 total)")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Feature files not found in {self.model_dir}. {e}")
    
    def _preprocess_sequence(self, sequence):
        """Convert peptide sequence to all 986 Mordred descriptors"""
        # Convert to SMILES
        smiles = self.fasta_parser.peptide_to_smiles(sequence)
        
        # Extract all 986 Mordred descriptors (with KNN imputation)
        descriptors_df = self.feature_extractor.extract_features(smiles)
        
        return descriptors_df  # Returns all 986 features
    
    def _preprocess_smiles(self, smiles):
        """Convert SMILES string directly to all 986 Mordred descriptors"""
        # Extract all 986 Mordred descriptors (with KNN imputation)
        descriptors_df = self.feature_extractor.extract_features(smiles)
        
        return descriptors_df  # Returns all 986 features
    
    def predict_single(self, sequence):
        """Predict CPP activity for a single peptide sequence"""
        # Extract all 986 Mordred descriptors
        all_descriptors_df = self._preprocess_sequence(sequence)
        
        # Select specific features for each model
        xgb_features_data = all_descriptors_df[self.xgb_features].values.reshape(1, -1)
        lgb_features_data = all_descriptors_df[self.lgb_features].values.reshape(1, -1)
        
        # Get predictions from both models
        xgb_proba = self.xgb_model.predict_proba(xgb_features_data)
        lgb_proba = self.lgb_model.predict_proba(lgb_features_data)
        
        # Extract positive class probabilities
        xgb_prob = xgb_proba[0, 1]
        lgb_prob = lgb_proba[0, 1]
        
        # Soft voting ensemble (average probabilities)
        ensemble_proba = np.mean([xgb_proba, lgb_proba], axis=0)
        ensemble_prob = ensemble_proba[0, 1]
        prediction = np.argmax(ensemble_proba, axis=1)[0]
        
        return {
            'sequence': sequence,
            'probability': ensemble_prob,
            'prediction': prediction,
            'xgb_probability': xgb_prob,
            'lgb_probability': lgb_prob
        }
    
    def predict_smiles(self, smiles):
        """Predict CPP activity for a single SMILES string"""
        # Extract all 986 Mordred descriptors
        all_descriptors_df = self._preprocess_smiles(smiles)
        
        # Select specific features for each model
        xgb_features_data = all_descriptors_df[self.xgb_features].values.reshape(1, -1)
        lgb_features_data = all_descriptors_df[self.lgb_features].values.reshape(1, -1)
        
        # Get predictions from both models
        xgb_proba = self.xgb_model.predict_proba(xgb_features_data)
        lgb_proba = self.lgb_model.predict_proba(lgb_features_data)
        
        # Extract positive class probabilities
        xgb_prob = xgb_proba[0, 1]
        lgb_prob = lgb_proba[0, 1]
        
        # Soft voting ensemble (average probabilities)
        ensemble_proba = np.mean([xgb_proba, lgb_proba], axis=0)
        ensemble_prob = ensemble_proba[0, 1]
        prediction = np.argmax(ensemble_proba, axis=1)[0]
        
        return {
            'smiles': smiles,
            'probability': ensemble_prob,
            'prediction': prediction,
            'xgb_probability': xgb_prob,
            'lgb_probability': lgb_prob
        }
    
    def predict_fasta_file(self, fasta_file):
        """Predict CPP activity for sequences in a FASTA file"""
        sequences = self.fasta_parser.parse_fasta_file(fasta_file)
        results = []
        
        for seq_id, sequence in sequences.items():
            try:
                result = self.predict_single(sequence)
                result['sequence_id'] = seq_id
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not process sequence {seq_id}: {e}")
                
        return pd.DataFrame(results)
    
    def predict_batch(self, sequences):
        """Predict CPP activity for a list of sequences (more efficient for large batches)"""
        results = []
        
        for i, sequence in enumerate(sequences):
            try:
                result = self.predict_single(sequence)
                result['sequence_id'] = f"seq_{i+1}"
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not process sequence {i+1}: {e}")
                
        return pd.DataFrame(results)