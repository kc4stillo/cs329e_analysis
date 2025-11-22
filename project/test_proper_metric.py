#!/usr/bin/env python3
"""
Proper implementation test following polarization_metric.md specification exactly.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ProperPolarizationMetricTester:
    """Exact implementation following polarization_metric.md specification."""
    
    def create_test_scenarios(self):
        """Create test scenarios for validation."""
        scenarios = {}
        np.random.seed(42)
        
        # 1. Echo Chamber: 85% vs 15%
        echo_data, _ = make_blobs(n_samples=[850, 150], centers=[[0, 0], [5, 5]], 
                                 cluster_std=[0.4, 0.6], random_state=42)
        echo_labels = np.array([0]*850 + [1]*150)
        scenarios['echo_chamber'] = (echo_data, echo_labels)
        
        # 2. Binary Polarization: 50% vs 50%
        binary_data, _ = make_blobs(n_samples=[500, 500], centers=[[0, 0], [8, 0]], 
                                   cluster_std=[1.0, 1.0], random_state=43)
        binary_labels = np.array([0]*500 + [1]*500)
        scenarios['binary_polarization'] = (binary_data, binary_labels)
        
        # 3. Multi-Polar: 4 balanced groups
        multi_data, _ = make_blobs(n_samples=[250, 250, 250, 250], centers=[[0, 0], [6, 0], [0, 6], [6, 6]], 
                                  cluster_std=[0.8, 0.8, 0.8, 0.8], random_state=44)
        multi_labels = np.array([0]*250 + [1]*250 + [2]*250 + [3]*250)
        scenarios['multi_polar'] = (multi_data, multi_labels)
        
        # 4. Fragmentation: 7 groups
        frag_centers = [[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3], [4, 6]]
        frag_samples = [140, 140, 140, 140, 150, 150, 140]
        frag_data, _ = make_blobs(n_samples=frag_samples, centers=frag_centers,
                                 cluster_std=[0.9]*7, random_state=45)
        frag_labels = []
        for i, n in enumerate(frag_samples):
            frag_labels.extend([i] * n)
        frag_labels = np.array(frag_labels)
        scenarios['fragmentation'] = (frag_data, frag_labels)
        
        # 5. Consensus: Single group
        consensus_data, _ = make_blobs(n_samples=1000, centers=[[0, 0]], 
                                      cluster_std=1.2, random_state=46)
        consensus_labels = np.array([0] * 1000)
        scenarios['consensus'] = (consensus_data, consensus_labels)
        
        return scenarios
    
    def entropy_component(self, cluster_sizes: np.ndarray) -> float:
        """
        Entropy-Based Fragmentation Score - EXACT from polarization_metric.md lines 90-99
        """
        if len(cluster_sizes) <= 1:
            return 0.0
            
        probs = cluster_sizes / np.sum(cluster_sizes)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(cluster_sizes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Group amplification: more groups = higher polarization potential
        group_factor = min((len(cluster_sizes) / 7) ** 1.0, 1.0)
        return normalized_entropy * group_factor  # [0,1]
    
    def distribution_component(self, cluster_sizes: np.ndarray) -> float:
        """
        Distribution-Based Concentration Score - EXACT from polarization_metric.md lines 103-107
        """
        if len(cluster_sizes) <= 1:
            return 0.0
            
        probs = cluster_sizes / np.sum(cluster_sizes)
        concentration = np.sum(probs**2)  # Herfindahl index
        return 1 - concentration  # Inverted for polarization [0,1]
    
    def quality_component(self, quality_score: float, n_groups: int) -> float:
        """
        Quality-Complexity Score - EXACT from polarization_metric.md lines 111-114
        """
        complexity_bonus = 1 + 0.2 * np.log(n_groups) / np.log(7) if n_groups > 2 else 1
        return quality_score * complexity_bonus  # Can exceed 1, normalized later
    
    def detect_echo_chamber(self, cluster_sizes: np.ndarray, embeddings: np.ndarray, 
                           labels: np.ndarray) -> Tuple[bool, float]:
        """
        Multi-criteria echo chamber detection following polarization_metric.md specification.
        Implements dominance + homogeneity + isolation analysis.
        """
        if len(cluster_sizes) < 2:
            return False, 0.0
            
        # 1. Dominance check: >85% in single group
        max_proportion = np.max(cluster_sizes) / np.sum(cluster_sizes)
        dominance_score = max(0, (max_proportion - 0.85) / 0.15) if max_proportion > 0.85 else 0.0
        
        if dominance_score == 0.0:
            return False, 0.0
            
        # 2. Internal homogeneity: Low variance within dominant group
        dominant_idx = np.argmax(cluster_sizes)
        dominant_mask = labels == dominant_idx
        
        if np.sum(dominant_mask) < 2:
            return False, 0.0
            
        dominant_embeddings = embeddings[dominant_mask]
        intra_variance = np.mean(np.var(dominant_embeddings, axis=0))
        
        # Normalize variance by data scale
        global_variance = np.mean(np.var(embeddings, axis=0))
        homogeneity_score = 1 - (intra_variance / (global_variance + 1e-10))
        homogeneity_score = max(0, min(1, homogeneity_score))
        
        # 3. Isolation analysis: High separation between dominant and minority groups
        minority_mask = ~dominant_mask
        if np.sum(minority_mask) == 0:
            return True, dominance_score  # Only dominant group exists
            
        minority_embeddings = embeddings[minority_mask]
        
        # Calculate center distances
        dominant_center = np.mean(dominant_embeddings, axis=0)
        minority_center = np.mean(minority_embeddings, axis=0)
        center_distance = np.linalg.norm(dominant_center - minority_center)
        
        # Normalize by average intra-cluster distances
        avg_intra_dist = (np.mean([np.linalg.norm(p - dominant_center) for p in dominant_embeddings]) +
                         np.mean([np.linalg.norm(p - minority_center) for p in minority_embeddings])) / 2
        
        isolation_score = center_distance / (avg_intra_dist + 1e-10)
        isolation_score = min(isolation_score / 3.0, 1.0)  # Normalize to [0,1]
        
        # Combined confidence - weights from polarization_metric.md context
        confidence = (dominance_score * 0.5 + homogeneity_score * 0.3 + isolation_score * 0.2)
        is_echo = confidence > 0.5
        
        return is_echo, confidence
    
    def echo_chamber_penalty(self, cluster_sizes: np.ndarray, embeddings: np.ndarray, 
                           labels: np.ndarray) -> float:
        """
        Echo Chamber Detection and Penalty - EXACT from polarization_metric.md lines 117-125
        """
        # Multi-criteria detection: dominance + homogeneity + isolation
        is_echo, confidence = self.detect_echo_chamber(cluster_sizes, embeddings, labels)
        if is_echo and confidence > 0.5:
            penalty_factor = confidence * 0.8  # Up to 80% reduction
            return 1 - penalty_factor
        return 1.0
    
    def calculate_quality_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate base quality score using silhouette and CH scores."""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
            
        try:
            silhouette = silhouette_score(embeddings, labels)
            ch_score = calinski_harabasz_score(embeddings, labels)
            
            # Normalize silhouette to [0,1]
            silhouette_norm = (silhouette + 1) / 2
            
            # Normalize CH score (empirical scaling)
            ch_norm = min(ch_score / 1000, 1.0)
            
            # Combine scores
            quality = (silhouette_norm * 0.7 + ch_norm * 0.3)
            return max(0, min(1, quality))
            
        except Exception:
            return 0.0
    
    def integrated_polarization_index(self, cluster_sizes: np.ndarray, embeddings: np.ndarray,
                                     labels: np.ndarray, consensus_confidence: float = 0.8) -> Dict[str, Any]:
        """
        Final Bounded Integration - EXACT from polarization_metric.md lines 129-156
        """
        if len(cluster_sizes) <= 1:
            return {
                'polarization_index': 0.0,
                'entropy': 0.0,
                'distribution': 0.0,
                'quality_raw': 0.0,
                'quality_adjusted': 0.0,
                'is_echo_chamber': False,
                'echo_confidence': 0.0,
                'echo_penalty': 1.0,
                'base_score': 0.0,
                'base_polarization': 0.0,
                'weights': [0.0, 0.0, 0.0],
                'reliability_factor': 0.8 + 0.2 * consensus_confidence
            }
        
        # Calculate components
        entropy = self.entropy_component(cluster_sizes)
        distribution = self.distribution_component(cluster_sizes) 
        quality_score = self.calculate_quality_score(embeddings, labels)
        quality = self.quality_component(quality_score, len(cluster_sizes))
        
        # Adaptive weighting based on group context - EXACT from lines 136-141
        if len(cluster_sizes) == 2:
            weights = [0.3, 0.5, 0.2]  # Emphasize distribution for binary
        elif len(cluster_sizes) <= 4:
            weights = [0.5, 0.3, 0.2]  # Emphasize entropy for multi-polar
        else:
            weights = [0.6, 0.2, 0.2]  # Strongly emphasize entropy for fragmentation
        
        # Base combination - line 144
        base_score = (weights[0] * entropy + weights[1] * distribution + weights[2] * quality)
        
        # Reliability scaling - lines 147-148
        reliability_factor = 0.8 + 0.2 * consensus_confidence
        base_polarization = base_score * reliability_factor
        
        # Echo chamber adjustment - lines 151-152
        echo_penalty = self.echo_chamber_penalty(cluster_sizes, embeddings, labels)
        final_score = base_polarization * echo_penalty
        
        # Critical: Ensure [0,1] bounds - line 155
        final_polarization = np.clip(final_score, 0, 1)
        
        # Get echo chamber details for analysis
        is_echo, echo_confidence = self.detect_echo_chamber(cluster_sizes, embeddings, labels)
        
        return {
            'polarization_index': final_polarization,
            'entropy': entropy,
            'distribution': distribution,
            'quality_raw': quality_score,
            'quality_adjusted': quality,
            'is_echo_chamber': is_echo,
            'echo_confidence': echo_confidence,
            'echo_penalty': echo_penalty,
            'base_score': base_score,
            'base_polarization': base_polarization,
            'weights': weights,
            'reliability_factor': reliability_factor
        }
    
    def run_proper_test(self):
        """Run the proper polarization metric test."""
        scenarios = self.create_test_scenarios()
        
        print("PROPER POLARIZATION METRIC IMPLEMENTATION TEST")
        print("Following polarization_metric.md specification EXACTLY")
        print("=" * 70)
        
        results = []
        targets = {
            'consensus': (0.0, 0.2),
            'echo_chamber': (0.05, 0.25),  # Updated based on spec lines 172-176
            'binary_polarization': (0.4, 0.6),
            'multi_polar': (0.6, 0.8),
            'fragmentation': (0.8, 1.0)
        }
        
        for name, (data, labels) in scenarios.items():
            cluster_sizes = np.array([np.sum(labels == i) for i in np.unique(labels)])
            
            # Use consensus confidence from spec (default 0.8)
            consensus_confidence = 0.8
            
            result = self.integrated_polarization_index(cluster_sizes, data, labels, consensus_confidence)
            
            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  Polarization Index: {result['polarization_index']:.3f}")
            print(f"  Components:")
            print(f"    Entropy: {result['entropy']:.3f}")
            print(f"    Distribution: {result['distribution']:.3f}")
            print(f"    Quality (raw): {result['quality_raw']:.3f}")
            print(f"    Quality (adjusted): {result['quality_adjusted']:.3f}")
            print(f"  Echo Chamber: {result['is_echo_chamber']} (conf: {result['echo_confidence']:.3f})")
            print(f"  Weights: {result['weights']}")
            print(f"  Base Score: {result['base_score']:.3f}")
            print(f"  Base Polarization: {result['base_polarization']:.3f}")
            print(f"  Echo Penalty: {result['echo_penalty']:.3f}")
            
            # Validate against targets
            score = result['polarization_index']
            if name in targets:
                min_t, max_t = targets[name]
                in_range = min_t <= score <= max_t
                status = "✅" if in_range else "❌"
                print(f"  Target: {min_t}-{max_t} {status}")
                
                results.append({
                    'name': name,
                    'score': score,
                    'in_range': in_range,
                    'details': result
                })
        
        # Summary
        passed = sum(r['in_range'] for r in results)
        score_range = max(r['score'] for r in results) - min(r['score'] for r in results)
        
        print(f"\n" + "=" * 70)
        print(f"PROPER IMPLEMENTATION RESULTS:")
        print(f"Scenarios Passed: {passed}/5")
        print(f"Discrimination Range: {score_range:.3f}")
        print(f"Expected Performance (from spec lines 172-176):")
        print(f"  Echo Chamber: 0.088, Binary: 0.466, Multi-polar: 0.649, Fragmentation: 0.889")
        print(f"  Expected Range: 0.800")
        
        if passed >= 4:
            print(f"\n✅ PROPER IMPLEMENTATION VALIDATED")
            print("This follows the research-backed integrated method from polarization_metric.md")
        else:
            print(f"\n❌ IMPLEMENTATION NEEDS ADJUSTMENT")
            print("May need parameter tuning to match expected performance")
        
        return results, passed >= 4
    
    def validate_expected_performance(self, results):
        """Compare against expected performance from spec lines 172-176."""
        expected = {
            'echo_chamber': 0.088,
            'binary_polarization': 0.466,
            'multi_polar': 0.649,
            'fragmentation': 0.889
        }
        
        print(f"\n" + "=" * 70)
        print("COMPARISON WITH EXPECTED PERFORMANCE:")
        
        for result in results:
            name = result['name']
            actual = result['score']
            if name in expected:
                expect = expected[name]
                diff = abs(actual - expect)
                print(f"{name.replace('_', ' ').title()}:")
                print(f"  Expected: {expect:.3f}, Actual: {actual:.3f}, Diff: {diff:.3f}")


def main():
    """Run the proper implementation test."""
    tester = ProperPolarizationMetricTester()
    
    print("Testing proper implementation of polarization_metric.md specification")
    print("This implements the research-backed integrated method with:")
    print("- Group amplification factor for entropy")
    print("- Complexity bonus for quality component") 
    print("- Multi-criteria echo chamber detection")
    print("- Exact adaptive weighting scheme")
    print("- Reliability factor based on consensus confidence")
    print()
    
    results, success = tester.run_proper_test()
    tester.validate_expected_performance(results)
    
    if success:
        print(f"\n🎉 PROPER POLARIZATION METRIC IMPLEMENTATION VALIDATED!")
    else:
        print(f"\n⚠️  Implementation needs parameter tuning")


if __name__ == "__main__":
    main()