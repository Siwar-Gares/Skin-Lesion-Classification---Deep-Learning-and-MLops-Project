"""
Data drift detection for model monitoring
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """
    Detects data drift using statistical tests
    """
    
    def __init__(self, reference_data: pd.DataFrame = None, threshold: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference dataset (training data)
            threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_report = {}
        
    def kolmogorov_smirnov_test(
        self, 
        reference: np.ndarray, 
        current: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Perform Kolmogorov-Smirnov test for distribution drift
        
        Args:
            reference: Reference data distribution
            current: Current data distribution
            
        Returns:
            Tuple of (statistic, p_value, is_drift_detected)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        is_drift = p_value < self.threshold
        
        return statistic, p_value, is_drift
    
    def chi_square_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> Tuple[float, float, bool]:
        """
        Perform Chi-square test for categorical drift
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins for histogram
            
        Returns:
            Tuple of (statistic, p_value, is_drift_detected)
        """
        # Create histograms
        ref_hist, bin_edges = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Avoid division by zero
        ref_hist = ref_hist + 1e-10
        curr_hist = curr_hist + 1e-10
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        # Chi-square test
        statistic = np.sum((ref_hist - curr_hist) ** 2 / ref_hist)
        df = bins - 1
        p_value = 1 - stats.chi2.cdf(statistic, df)
        is_drift = p_value < self.threshold
        
        return statistic, p_value, is_drift
    
    def detect_feature_drift(
        self,
        feature_name: str,
        reference_values: np.ndarray,
        current_values: np.ndarray,
        feature_type: str = 'numerical'
    ) -> Dict:
        """
        Detect drift for a single feature
        
        Args:
            feature_name: Name of the feature
            reference_values: Reference feature values
            current_values: Current feature values
            feature_type: 'numerical' or 'categorical'
            
        Returns:
            Dictionary with drift detection results
        """
        if feature_type == 'numerical':
            ks_stat, p_value, is_drift = self.kolmogorov_smirnov_test(
                reference_values, current_values
            )
            
            result = {
                'feature': feature_name,
                'type': feature_type,
                'test': 'kolmogorov_smirnov',
                'statistic': float(ks_stat),
                'p_value': float(p_value),
                'drift_detected': bool(is_drift),
                'reference_mean': float(np.mean(reference_values)),
                'current_mean': float(np.mean(current_values)),
                'reference_std': float(np.std(reference_values)),
                'current_std': float(np.std(current_values))
            }
        else:
            chi_stat, p_value, is_drift = self.chi_square_test(
                reference_values, current_values
            )
            
            result = {
                'feature': feature_name,
                'type': feature_type,
                'test': 'chi_square',
                'statistic': float(chi_stat),
                'p_value': float(p_value),
                'drift_detected': bool(is_drift)
            }
        
        return result
    
    def detect_dataset_drift(
        self,
        current_data: pd.DataFrame,
        feature_types: Dict[str, str] = None
    ) -> Dict:
        """
        Detect drift across entire dataset
        
        Args:
            current_data: Current dataset to compare
            feature_types: Dictionary mapping feature names to types
            
        Returns:
            Comprehensive drift report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        drift_results = []
        features_with_drift = []
        
        common_features = set(self.reference_data.columns) & set(current_data.columns)
        
        for feature in common_features:
            ref_values = self.reference_data[feature].dropna().values
            curr_values = current_data[feature].dropna().values
            
            # Determine feature type
            if feature_types and feature in feature_types:
                ftype = feature_types[feature]
            else:
                ftype = 'numerical' if np.issubdtype(ref_values.dtype, np.number) else 'categorical'
            
            result = self.detect_feature_drift(
                feature, ref_values, curr_values, ftype
            )
            drift_results.append(result)
            
            if result['drift_detected']:
                features_with_drift.append(feature)
        
        # Summary
        self.drift_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_features': len(common_features),
            'features_with_drift': len(features_with_drift),
            'drift_percentage': len(features_with_drift) / len(common_features) * 100 if common_features else 0,
            'threshold': self.threshold,
            'drifted_features': features_with_drift,
            'detailed_results': drift_results
        }
        
        return self.drift_report
    
    def save_report(self, filepath: str):
        """Save drift report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.drift_report, f, indent=2)
        print(f"✓ Drift report saved to {filepath}")
    
    def get_summary(self) -> str:
        """Get human-readable summary of drift detection"""
        if not self.drift_report:
            return "No drift detection performed yet"
        
        summary = f"""
Drift Detection Summary
{'='*50}
Timestamp: {self.drift_report['timestamp']}
Total Features Analyzed: {self.drift_report['total_features']}
Features with Drift: {self.drift_report['features_with_drift']}
Drift Percentage: {self.drift_report['drift_percentage']:.2f}%
Threshold (p-value): {self.drift_report['threshold']}

Drifted Features:
{', '.join(self.drift_report['drifted_features']) if self.drift_report['drifted_features'] else 'None'}
"""
        return summary
