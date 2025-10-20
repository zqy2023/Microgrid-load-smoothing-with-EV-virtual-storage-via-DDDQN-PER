#!/usr/bin/env python3
"""
数据完整性检查工具
验证数据集是否符合发表标准
"""

import pandas as pd
import numpy as np
import json
import yaml
import hashlib
import os
from datetime import datetime
from pathlib import Path

class DataIntegrityChecker:
    def __init__(self, base_path="./"):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data_processed"
        self.checks_path = self.base_path / "data_checks"
        self.results = {}
        
    def calculate_file_hash(self, filepath):
        """计算文件SHA256哈希值"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def load_expectations(self, dataset_name):
        """加载数据验证规则"""
        expectations_file = self.checks_path / f"{dataset_name}_expectations.yml"
        with open(expectations_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def validate_price_data(self):
        """验证OpenNEM价格数据"""
        print("🔍 验证OpenNEM价格数据...")
        
        # 加载期望和数据
        expectations = self.load_expectations('price')
        df = pd.read_parquet(self.data_path / 'price_15min.parquet')
        
        results = {
            'dataset': 'price_15min',
            'validation_timestamp': datetime.now().isoformat(),
            'file_hash': self.calculate_file_hash(self.data_path / 'price_15min.parquet'),
            'tests': {}
        }
        
        # 测试1: 数据完整性
        expected_rows = expectations['expectations']['data_completeness']['expected_rows']
        actual_rows = len(df)
        coverage_percent = (actual_rows / expected_rows) * 100
        
        results['tests']['data_completeness'] = {
            'expected_rows': expected_rows,
            'actual_rows': actual_rows,
            'coverage_percent': coverage_percent,
            'passed': coverage_percent >= 99.9
        }
        
        # 测试2: 时间完整性
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        is_monotonic = df['timestamp'].is_monotonic_increasing
        has_duplicates = df['timestamp'].duplicated().any()
        
        results['tests']['temporal_integrity'] = {
            'monotonic_timestamps': is_monotonic,
            'no_duplicates': not has_duplicates,
            'passed': is_monotonic and not has_duplicates
        }
        
        # 测试3: 价格特征
        price_stats = {
            'min_price': float(df['price'].min()),
            'max_price': float(df['price'].max()),
            'mean_price': float(df['price'].mean()),
            'std_price': float(df['price'].std()),
            'negative_price_percent': float((df['price'] < 0).sum() / len(df) * 100),
            'high_price_percent': float((df['price'] > 300).sum() / len(df) * 100)
        }
        
        price_range = expectations['expectations']['price_characteristics']['expected_range']
        stats_range = expectations['expectations']['price_characteristics']['statistical_expectations']
        
        price_tests = {
            'price_range_valid': price_range['min_price'] <= price_stats['min_price'] and price_stats['max_price'] <= price_range['max_price'],
            'mean_in_range': stats_range['mean_price_range'][0] <= price_stats['mean_price'] <= stats_range['mean_price_range'][1],
            'std_in_range': stats_range['std_dev_range'][0] <= price_stats['std_price'] <= stats_range['std_dev_range'][1],
            'negative_prices_realistic': price_stats['negative_price_percent'] <= expectations['expectations']['price_characteristics']['negative_price_occurrence']['max_percent']
        }
        
        results['tests']['price_characteristics'] = {
            'statistics': price_stats,
            'validations': price_tests,
            'passed': all(price_tests.values())
        }
        
        # 测试4: 数据质量
        quality_tests = {
            'no_null_values': df['price'].isnull().sum() == 0,
            'no_infinite_values': not np.isinf(df['price']).any(),
            'no_duplicate_timestamps': not has_duplicates
        }
        
        results['tests']['data_quality'] = {
            'validations': quality_tests,
            'passed': all(quality_tests.values())
        }
        
        # 总体通过率
        passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
        total_tests = len(results['tests'])
        results['overall_pass_rate'] = passed_tests / total_tests
        results['ieee_compliant'] = results['overall_pass_rate'] >= 0.9
        
        self.results['price_data'] = results
        return results
    
    def validate_load_data(self):
        """验证Ausgrid负荷数据"""
        print("🔍 验证Ausgrid负荷数据...")
        
        expectations = self.load_expectations('load')
        df = pd.read_parquet(self.data_path / 'load_15min.parquet')
        
        results = {
            'dataset': 'load_15min',
            'validation_timestamp': datetime.now().isoformat(),
            'file_hash': self.calculate_file_hash(self.data_path / 'load_15min.parquet'),
            'tests': {}
        }
        
        # 测试1: 数据完整性
        expected_rows = expectations['expectations']['data_completeness']['expected_rows']
        actual_rows = len(df)
        
        results['tests']['data_completeness'] = {
            'expected_rows': expected_rows,
            'actual_rows': actual_rows,
            'passed': actual_rows == expected_rows
        }
        
        # 测试2: 负荷特征
        load_stats = {
            'min_load': float(df['load'].min()),
            'max_load': float(df['load'].max()),
            'mean_load': float(df['load'].mean()),
            'std_load': float(df['load'].std()),
            'peak_valley_ratio': float(df['load'].max() / df['load'].min()) if df['load'].min() > 0 else float('inf')
        }
        
        load_range = expectations['expectations']['load_characteristics']['expected_range']
        stats_range = expectations['expectations']['load_characteristics']['statistical_expectations']
        
        load_tests = {
            'load_range_valid': load_range['min_load'] <= load_stats['min_load'] and load_stats['max_load'] <= load_range['max_load'],
            'mean_in_range': stats_range['mean_load_range'][0] <= load_stats['mean_load'] <= stats_range['mean_load_range'][1],
            'positive_values_only': load_stats['min_load'] >= 0
        }
        
        results['tests']['load_characteristics'] = {
            'statistics': load_stats,
            'validations': load_tests,
            'passed': all(load_tests.values())
        }
        
        # 测试3: 数据质量
        quality_tests = {
            'no_null_values': df['load'].isnull().sum() == 0,
            'no_negative_values': (df['load'] < 0).sum() == 0,
            'no_infinite_values': not np.isinf(df['load']).any()
        }
        
        results['tests']['data_quality'] = {
            'validations': quality_tests,
            'passed': all(quality_tests.values())
        }
        
        # 总体通过率
        passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
        total_tests = len(results['tests'])
        results['overall_pass_rate'] = passed_tests / total_tests
        results['ieee_compliant'] = results['overall_pass_rate'] >= 0.9
        
        self.results['load_data'] = results
        return results
    
    def validate_ev_data(self):
        """验证CSIRO EV需求数据"""
        print("🔍 验证CSIRO EV需求数据...")
        
        expectations = self.load_expectations('ev')
        df = pd.read_parquet(self.data_path / 'ev_demand_15min.parquet')
        
        results = {
            'dataset': 'ev_demand_15min',
            'validation_timestamp': datetime.now().isoformat(),
            'file_hash': self.calculate_file_hash(self.data_path / 'ev_demand_15min.parquet'),
            'tests': {}
        }
        
        # 测试1: 数据完整性
        expected_rows = expectations['expectations']['data_completeness']['expected_rows']
        actual_rows = len(df)
        
        results['tests']['data_completeness'] = {
            'expected_rows': expected_rows,
            'actual_rows': actual_rows,
            'passed': actual_rows == expected_rows
        }
        
        # 测试2: EV需求特征
        ev_stats = {
            'min_demand': float(df['ev_demand'].min()),
            'max_demand': float(df['ev_demand'].max()),
            'mean_demand': float(df['ev_demand'].mean()),
            'std_demand': float(df['ev_demand'].std())
        }
        
        ev_range = expectations['expectations']['ev_demand_characteristics']['expected_range']
        stats_range = expectations['expectations']['ev_demand_characteristics']['statistical_expectations']
        
        ev_tests = {
            'demand_range_valid': ev_range['min_demand'] <= ev_stats['min_demand'] and ev_stats['max_demand'] <= ev_range['max_demand'],
            'mean_in_range': stats_range['mean_demand_range'][0] <= ev_stats['mean_demand'] <= stats_range['mean_demand_range'][1],
            'non_negative_values': ev_stats['min_demand'] >= 0
        }
        
        results['tests']['ev_characteristics'] = {
            'statistics': ev_stats,
            'validations': ev_tests,
            'passed': all(ev_tests.values())
        }
        
        # 测试3: 数据质量
        quality_tests = {
            'no_null_values': df['ev_demand'].isnull().sum() == 0,
            'no_negative_values': (df['ev_demand'] < 0).sum() == 0,
            'no_infinite_values': not np.isinf(df['ev_demand']).any()
        }
        
        results['tests']['data_quality'] = {
            'validations': quality_tests,
            'passed': all(quality_tests.values())
        }
        
        # 总体通过率
        passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
        total_tests = len(results['tests'])
        results['overall_pass_rate'] = passed_tests / total_tests
        results['ieee_compliant'] = results['overall_pass_rate'] >= 0.9
        
        self.results['ev_data'] = results
        return results
    
    def validate_soh_params(self):
        """验证SOH参数文献对齐"""
        print("🔍 验证SOH参数文献对齐...")
        
        with open(self.data_path / 'soh_params.json', 'r', encoding='utf-8') as f:
            soh_params = json.load(f)
        
        results = {
            'dataset': 'soh_params',
            'validation_timestamp': datetime.now().isoformat(),
            'file_hash': self.calculate_file_hash(self.data_path / 'soh_params.json'),
            'tests': {}
        }
        
        # 必需字段检查
        required_fields = ['parameters', 'formula', 'reference', 'units', 'literature_alignment']
        missing_fields = [field for field in required_fields if field not in soh_params]
        
        results['tests']['structure'] = {
            'required_fields_present': len(missing_fields) == 0,
            'missing_fields': missing_fields,
            'passed': len(missing_fields) == 0
        }
        
        # 参数值检查
        if 'parameters' in soh_params:
            params = soh_params['parameters']
            param_tests = {
                'has_a_parameter': 'a' in params,
                'has_b_parameter': 'b' in params, 
                'has_c_parameter': 'c' in params,
                'a_in_range': 0.0001 <= params.get('a', 0) <= 0.001,
                'b_in_range': 1e-6 <= params.get('b', 0) <= 1e-4,
                'c_in_range': 0.0001 <= params.get('c', 0) <= 0.01
            }
        else:
            param_tests = {'parameters_missing': True}
        
        results['tests']['parameters'] = {
            'validations': param_tests,
            'passed': all(param_tests.values()) if 'parameters_missing' not in param_tests else False
        }
        
        # 文献对齐检查
        literature_tests = {
            'has_primary_source': 'literature_alignment' in soh_params and 'primary_source' in soh_params.get('literature_alignment', {}),
            'has_ieee_compliance': 'ieee_compliance' in soh_params,
            'peer_reviewed': soh_params.get('ieee_compliance', {}).get('peer_reviewed_source', False)
        }
        
        results['tests']['literature_alignment'] = {
            'validations': literature_tests,
            'passed': all(literature_tests.values())
        }
        
        # 总体通过率
        passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
        total_tests = len(results['tests'])
        results['overall_pass_rate'] = passed_tests / total_tests
        results['ieee_compliant'] = results['overall_pass_rate'] >= 0.9
        
        self.results['soh_params'] = results
        return results
    
    def generate_sha256_checksums(self):
        """生成所有数据文件的SHA256校验和"""
        print("🔐 生成数据文件校验和...")
        
        data_files = [
            'price_15min.parquet',
            'load_15min.parquet', 
            'ev_demand_15min.parquet',
            'soh_params.json'
        ]
        
        checksums = {}
        for filename in data_files:
            filepath = self.data_path / filename
            if filepath.exists():
                checksums[filename] = self.calculate_file_hash(filepath)
        
        # 保存到文件
        checksum_file = self.base_path / 'sha256sum.txt'
        with open(checksum_file, 'w') as f:
            f.write("# IEEE Access Publication Data Integrity Checksums\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("# Verify with: sha256sum -c sha256sum.txt\n\n")
            for filename, checksum in checksums.items():
                f.write(f"{checksum}  data_processed/{filename}\n")
        
        return checksums
    
    def run_all_validations(self):
        """运行所有数据验证"""
        print("🚀 开始IEEE Access发表标准数据验证\n")
        
        # 验证各数据集
        self.validate_price_data()
        self.validate_load_data()
        self.validate_ev_data()
        self.validate_soh_params()
        
        # 生成校验和
        checksums = self.generate_sha256_checksums()
        
        # 生成总体报告
        overall_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'datasets_validated': len(self.results),
            'ieee_compliance_summary': {},
            'data_integrity': checksums,
            'detailed_results': self.results
        }
        
        # 计算总体合规性
        for dataset, result in self.results.items():
            overall_results['ieee_compliance_summary'][dataset] = {
                'pass_rate': result['overall_pass_rate'],
                'ieee_compliant': result['ieee_compliant']
            }
        
        total_pass_rate = sum(r['overall_pass_rate'] for r in self.results.values()) / len(self.results)
        overall_results['total_pass_rate'] = total_pass_rate
        overall_results['publication_ready'] = total_pass_rate >= 0.8
        
        # 保存完整报告
        report_file = self.base_path / 'docs' / 'data_quality' / 'ieee_data_validation_report.json'
        os.makedirs(report_file.parent, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 IEEE Access数据验证摘要")
        print("="*60)
        
        for dataset, summary in overall_results['ieee_compliance_summary'].items():
            status = "✅ 合规" if summary['ieee_compliant'] else "❌ 不合规"
            print(f"{dataset:<20}: {summary['pass_rate']:.1%} {status}")
        
        print(f"\n总体通过率: {total_pass_rate:.1%}")
        status = "🎉 发表就绪" if overall_results['publication_ready'] else "⚠️ 需要改进"
        print(f"发表状态: {status}")
        print(f"\n📄 详细报告: {report_file}")
        print(f"🔐 数据校验和: sha256sum.txt")
        
        return overall_results

if __name__ == "__main__":
    checker = DataIntegrityChecker()
    results = checker.run_all_validations() 