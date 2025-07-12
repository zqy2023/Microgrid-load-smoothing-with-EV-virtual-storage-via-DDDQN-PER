#!/usr/bin/env python3
"""
负荷数据处理
处理Ausgrid家庭用电数据，包括采样、缺失值处理和15分钟聚合
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ausgrid_data(src_path: str) -> pd.DataFrame:
    """加载Ausgrid用电数据"""
    logger.info(f"加载数据: {src_path}")
    
    try:
        df = pd.read_csv(src_path, encoding='utf-8', low_memory=False)
        logger.info(f"加载记录数: {len(df):,}")
        return df
    except Exception as e:
        logger.error(f"加载失败: {e}")
        raise


def sample_customers(df: pd.DataFrame, n_customers: int = 300) -> pd.DataFrame:
    """Scientifically sample customer subset."""
    logger.info(f"Sampling {n_customers} customers from dataset")
    
    # Get unique customer IDs
    customer_col = None
    for col in ['Customer', 'customer_id', 'CUSTOMER_ID', 'ID']:
        if col in df.columns:
            customer_col = col
            break
    
    if customer_col is None:
        raise ValueError("No customer ID column found")
    
    unique_customers = df[customer_col].unique()
    logger.info(f"Found {len(unique_customers):,} unique customers")
    
    # Stratified sampling by consumption levels
    if 'General supply (kWh)' in df.columns:
        agg_col = 'General supply (kWh)'
    else:
        agg_col = df.select_dtypes(include=[np.number]).columns[0]
    
    customer_stats = df.groupby(customer_col).agg({
        agg_col: ['mean', 'std', 'count']
    }).reset_index()
    
    customer_stats.columns = [customer_col, 'mean_consumption', 'std_consumption', 'record_count']
    
    # Filter customers with sufficient data
    min_records = 100  # Minimum records per customer
    valid_customers = customer_stats[customer_stats['record_count'] >= min_records]
    
    if len(valid_customers) < n_customers:
        logger.warning(f"Only {len(valid_customers)} customers have sufficient data")
        n_customers = len(valid_customers)
    
    # Sample customers
    sampled_customers = valid_customers.sample(n=n_customers, random_state=42)
    selected_ids = sampled_customers[customer_col].tolist()
    
    # Filter original data
    sampled_df = df[df[customer_col].isin(selected_ids)].copy()
    
    logger.info(f"Sampled {len(sampled_df):,} records from {n_customers} customers")
    return sampled_df


def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Process datetime columns."""
    logger.info("Processing datetime information")
    
    # Find datetime column
    datetime_col = None
    for col in ['Reading date and time', 'datetime', 'timestamp', 'date']:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col is None:
        raise ValueError("No datetime column found")
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df[datetime_col])
    
    return df


def filter_time_window(df: pd.DataFrame, since: str, until: str) -> pd.DataFrame:
    """Filter data to specified time window."""
    logger.info(f"Filtering data from {since} to {until}")
    
    mask = (df['datetime'] >= since) & (df['datetime'] < until)
    filtered_df = df[mask].copy()
    
    logger.info(f"Filtered to {len(filtered_df):,} records")
    return filtered_df


def aggregate_load_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate load data to 15-minute intervals."""
    logger.info("Aggregating load data to 15-minute intervals")
    
    # Find consumption column
    consumption_col = None
    for col in ['General supply (kWh)', 'consumption', 'load', 'demand']:
        if col in df.columns:
            consumption_col = col
            break
    
    if consumption_col is None:
        # Use first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        consumption_col = numeric_cols[0]
    
    logger.info(f"Using consumption column: {consumption_col}")
    
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Aggregate by 15-minute intervals (sum across all customers)
    aggregated = df.groupby(pd.Grouper(freq='15min'))[consumption_col].sum()
    
    # Handle missing values
    aggregated = aggregated.fillna(method='ffill').fillna(method='bfill')
    
    # Create output dataframe
    result_df = pd.DataFrame({
        'datetime': aggregated.index,
        'load_kw': aggregated.values * 4  # Convert kWh to kW (15min -> 1h)
    })
    
    logger.info(f"Aggregated to {len(result_df):,} 15-minute intervals")
    return result_df


def validate_load_data(df: pd.DataFrame) -> dict:
    """Validate load data and return statistics."""
    logger.info("Validating load data")
    
    stats = {
        'total_records': len(df),
        'min_load': df['load_kw'].min(),
        'max_load': df['load_kw'].max(),
        'mean_load': df['load_kw'].mean(),
        'peak_valley_ratio': df['load_kw'].max() / df['load_kw'].min(),
        'missing_count': df['load_kw'].isna().sum(),
        'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}"
    }
    
    logger.info(f"Load statistics:")
    logger.info(f"  Records: {stats['total_records']:,}")
    logger.info(f"  Range: {stats['min_load']:.2f} to {stats['max_load']:.2f} kW")
    logger.info(f"  Mean: {stats['mean_load']:.2f} kW")
    logger.info(f"  Peak-valley ratio: {stats['peak_valley_ratio']:.2f}")
    logger.info(f"  Missing values: {stats['missing_count']:,}")
    
    return stats


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed data to parquet file."""
    logger.info(f"Saving processed data to: {output_path}")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df):,} records to {output_path}")


def main():
    """Main ETL pipeline."""
    parser = argparse.ArgumentParser(description="Process household load data")
    parser.add_argument("--src", required=True, help="Path to source Ausgrid CSV file")
    parser.add_argument("--customers", type=int, default=300, help="Number of customers to sample")
    parser.add_argument("--since", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", required=True, help="Output parquet file path")
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_ausgrid_data(args.src)
        
        # Sample customers
        df_sampled = sample_customers(df, args.customers)
        
        # Process datetime
        df_processed = process_datetime(df_sampled)
        
        # Filter time window
        df_filtered = filter_time_window(df_processed, args.since, args.until)
        
        # Aggregate load data
        df_aggregated = aggregate_load_data(df_filtered)
        
        # Validate data
        stats = validate_load_data(df_aggregated)
        
        # Save processed data
        save_processed_data(df_aggregated, args.out)
        
        logger.info("Load ETL pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 