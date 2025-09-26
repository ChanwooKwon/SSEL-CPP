"""
SSEL-CPP: Cell Penetrating Peptide Prediction Tool
User-friendly interface for predicting CPP activity from FASTA sequences
"""

import argparse
import sys
from pathlib import Path
from src.ssel_cpp import SSELCPP

def main():
    parser = argparse.ArgumentParser(
        description="SSEL-CPP: Predict Cell Penetrating Peptide activity from FASTA sequences, peptide sequences, or SMILES strings"
    )
    parser.add_argument(
        "input", 
        help="Input FASTA file, peptide sequence, or SMILES string"
    )
    parser.add_argument(
        "--output", "-o",
        default="predictions.csv",
        help="Output file for predictions (default: predictions.csv)"
    )
    parser.add_argument(
        "--sequence", "-s",
        action="store_true",
        help="Input is a single peptide sequence (not a file)"
    )
    parser.add_argument(
        "--smiles",
        action="store_true",
        help="Input is a SMILES string (not a peptide sequence or file)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize SSEL-CPP model
        model = SSELCPP()
        
        if args.smiles:
            # Single SMILES prediction
            print(f"Predicting CPP activity for SMILES: {args.input}")
            result = model.predict_smiles(args.input)
            print(f"\nPrediction Results:")
            print(f"CPP Probability: {result['probability']:.4f}")
            print(f"Classification: {'CPP' if result['prediction'] == 1 else 'Non-CPP'}")
            
        elif args.sequence:
            # Single sequence prediction
            print(f"Predicting CPP activity for sequence: {args.input}")
            result = model.predict_single(args.input)
            print(f"\nPrediction Results:")
            print(f"CPP Probability: {result['probability']:.4f}")
            print(f"Classification: {'CPP' if result['prediction'] == 1 else 'Non-CPP'}")
            
        else:
            # File-based prediction
            if not Path(args.input).exists():
                print(f"Error: File {args.input} not found!")
                sys.exit(1)
                
            print(f"Processing FASTA file: {args.input}")
            results = model.predict_fasta_file(args.input)
            
            # Save results
            results.to_csv(args.output, index=False)
            print(f"Predictions saved to: {args.output}")
            
            # Print summary
            cpp_count = (results['prediction'] == 1).sum()
            total_count = len(results)
            print(f"\nSummary:")
            print(f"Total sequences: {total_count}")
            print(f"Predicted CPPs: {cpp_count}")
            print(f"Predicted Non-CPPs: {total_count - cpp_count}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()