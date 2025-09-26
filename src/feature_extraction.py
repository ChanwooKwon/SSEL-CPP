"""
Mordred descriptor extraction for peptide sequences
"""

import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from mordred import Calculator, descriptors
from pathlib import Path

class MordredExtractor:
    def __init__(self, model_dir="models"):
        """Initialize Mordred calculator with pre-trained imputer and feature order"""
        self.calc = Calculator(descriptors, ignore_3D=True)
        self.model_dir = Path(model_dir)
        
        # Load pre-trained KNN imputer and feature order
        self._load_imputer()
        self._load_feature_order()
        
    def _load_imputer(self):
        """Load pre-trained KNN imputer"""
        try:
            self.imputer = joblib.load(self.model_dir / "knn_imputer.pkl")
            print("Pre-trained KNN imputer loaded successfully!")
        except FileNotFoundError:
            raise FileNotFoundError(f"Pre-trained KNN imputer not found in {self.model_dir}")
    
    def _load_feature_order(self):
        """Load the exact feature order used during training (986 features)"""
        try:
            # Load feature order - CSV starts directly with feature names (no header)
            with open(self.model_dir / "knn_feature_order.csv", 'r') as f:
                self.feature_order = [line.strip() for line in f.readlines()]
            print(f"Feature order loaded: {len(self.feature_order)} features")
        except FileNotFoundError:
            raise FileNotFoundError(f"Feature order file not found in {self.model_dir}")
        
    def extract_features(self, smiles):
        """Extract and process Mordred 2D descriptors from SMILES"""
        if isinstance(smiles, str):
            smiles = [smiles]
            
        mols = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            mols.append(mol)
        
        # Calculate ALL Mordred descriptors
        descriptors_df = self.calc.pandas(mols)
        
        # Handle missing values and infinite values
        descriptors_df = descriptors_df.replace([np.inf, -np.inf], np.nan)
        
        # IMPORTANT: Reorder columns to match training feature order (986 features)
        try:
            descriptors_df = descriptors_df[self.feature_order]
        except KeyError as e:
            missing_features = set(self.feature_order) - set(descriptors_df.columns)
            extra_features = set(descriptors_df.columns) - set(self.feature_order)
            raise ValueError(f"Feature mismatch! Missing: {missing_features}, Extra: {extra_features}")
        
        # Apply pre-trained KNN imputation (986 features → 986 features)
        descriptors_imputed = self.imputer.transform(descriptors_df)
        
        # Convert back to DataFrame with original column names in correct order
        descriptors_df = pd.DataFrame(
            descriptors_imputed, 
            columns=self.feature_order,  # All 986 features
            index=descriptors_df.index
        )
        
        return descriptors_df  # Returns 986 features after KNN imputation