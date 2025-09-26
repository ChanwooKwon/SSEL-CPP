"""
FASTA file parsing and peptide to SMILES conversion utilities
"""

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

class FastaParser:
    def __init__(self):
        """Initialize FASTA parser"""
        pass
    
    def parse_fasta_file(self, fasta_file):
        """Parse FASTA file and return dictionary of sequences"""
        sequences = {}
        
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences[record.id] = str(record.seq)
        except Exception as e:
            raise ValueError(f"Error parsing FASTA file: {e}")
        
        return sequences
    
    def peptide_to_smiles(self, peptide_sequence):
        """Convert peptide sequence to SMILES using RDKit"""
        try:
            # Use RDKit's peptide functionality
            mol = Chem.MolFromFASTA(peptide_sequence)
            if mol is None:
                # Alternative approach using sequence to SMILES
                mol = Chem.MolFromSequence(peptide_sequence)
            
            if mol is None:
                raise ValueError(f"Could not convert peptide sequence to molecule: {peptide_sequence}")
            
            smiles = Chem.MolToSmiles(mol)
            return smiles
            
        except Exception as e:
            raise ValueError(f"Error converting peptide to SMILES: {e}")
    
    def validate_sequence(self, sequence):
        """Validate peptide sequence"""
        sequence = sequence.upper().strip()
        
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')  # Standard amino acids
        for aa in sequence:
            if aa not in valid_aa:
                return False, f"Invalid amino acid: {aa}"
        
        return True, "Valid sequence"