#!/usr/bin/env python3
"""
EV Demand Data ETL Pipeline

Processes CSIRO EV uptake projections for specified regions and scenarios.
Generates synthetic EV charging demand profiles.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_csiro_projections(csiro_dir: str, state: str = "NSWACT") -> dict:
    """Load CSIRO EV fleet projections."""
    logger.info(f"Loading CSIRO projections for {state}")
    
    projections = {}
    csiro_path = Path(csiro_dir)
    
    # Load different vehicle types
    vehicle_types = ['BEV', 'HV', 'HYB', 'ICE', 'PHEV']
    
    for vtype in vehicle_types:
        file_pattern = f"FLEET_CONSUMPTION_PROJECTIONS_{vtype}_POSTCODE_{state}_*.csv"
        files = list(csiro_path.glob(file_pattern))
        
        if files:
            file_path = files[0]  # Use first matching file
            try:
                df = pd.read_csv(file_path)
                projections[vtype] = df
                logger.info(f"Loaded {vtype} projections: {len(df):,} records")
            except Exception as e:
                logger.warning(f"Failed to load {vtype} projections: {e}")
    
    return projections


def generate_ev_demand_profile(projections: dict, scenario: str = "MID", 
                             since: str = "2023-01-01", until: str = "2023-06-30") -> pd.DataFrame:
    """Generate EV charging demand profile."""
    logger.info(f"Generating EV demand profile for {scenario} scenario")
    
    # Focus on BEV and PHEV for charging demand
    ev_types = ['BEV', 'PHEV']
    total_evs = 0
    
    for ev_type in ev_types:
        if ev_type in projections:
            df = projections[ev_type]
            # Use middle scenario if available
            if scenario in df.columns:
                total_evs += df[scenario].sum()
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    total_evs += df[numeric_cols[0]].sum()
    
    logger.info(f"Total EVs for charging: {total_evs:,.0f}")
    
    # Generate time series
    start_date = pd.to_datetime(since)
    end_date = pd.to_datetime(until)
    time_range = pd.date_range(start=start_date, end=end_date, freq='15min')[:-1]  # Exclude end
    
    # Create synthetic charging patterns
    np.random.seed(42)  # Reproducible results
    
    demand_profile = []
    
    for timestamp in time_range:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday
        
        # Base demand factors
        base_demand = 0.1  # 10% of EVs charging at any time
        
        # Time-of-day patterns
        if 6 <= hour <= 9:  # Morning peak
            time_factor = 1.5
        elif 17 <= hour <= 21:  # Evening peak
            time_factor = 2.0
        elif 22 <= hour <= 6:  # Night charging
            time_factor = 0.8
        else:  # Off-peak
            time_factor = 0.6
        
        # Day-of-week patterns
        if day_of_week < 5:  # Weekdays
            day_factor = 1.2
        else:  # Weekends
            day_factor = 0.9
        
        # Seasonal factor (simplified)
        month = timestamp.month
        if month in [12, 1, 2]:  # Summer (Australia)
            seasonal_factor = 1.1
        elif month in [6, 7, 8]:  # Winter
            seasonal_factor = 0.9
        else:
            seasonal_factor = 1.0
        
        # Calculate demand
        demand_factor = base_demand * time_factor * day_factor * seasonal_factor
        
        # Add random variation
        noise = np.random.normal(0, 0.1)  # 10% noise
        demand_factor = max(0.01, demand_factor + noise)  # Minimum 1%
        
        # Convert to absolute demand (kW)
        # Assume average EV charging at 7.4 kW
        ev_demand_kw = total_evs * demand_factor * 7.4
        
        demand_profile.append(ev_demand_kw)
    
    # Create output dataframe
    result_df = pd.DataFrame({
        'datetime': time_range,
        'ev_demand_kw': demand_profile
    })
    
    logger.info(f"Generated {len(result_df):,} 15-minute EV demand records")
    return result_df


def validate_ev_data(df: pd.DataFrame) -> dict:
    """Validate EV demand data and return statistics."""
    logger.info("Validating EV demand data")
    
    stats = {
        'total_records': len(df),
        'min_demand': df['ev_demand_kw'].min(),
        'max_demand': df['ev_demand_kw'].max(),
        'mean_demand': df['ev_demand_kw'].mean(),
        'peak_valley_ratio': df['ev_demand_kw'].max() / df['ev_demand_kw'].min(),
        'missing_count': df['ev_demand_kw'].isna().sum(),
        'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}"
    }
    
    logger.info(f"EV demand statistics:")
    logger.info(f"  Records: {stats['total_records']:,}")
    logger.info(f"  Range: {stats['min_demand']:.2f} to {stats['max_demand']:.2f} kW")
    logger.info(f"  Mean: {stats['mean_demand']:.2f} kW")
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
    parser = argparse.ArgumentParser(description="Generate EV demand data from CSIRO projections")
    parser.add_argument("--csiro_dir", required=True, help="Path to CSIRO EV projections directory")
    parser.add_argument("--state", default="NSWACT", help="State/territory code")
    parser.add_argument("--scenario", default="MID", help="Scenario (LOW/MID/HIGH)")
    parser.add_argument("--since", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", required=True, help="Output parquet file path")
    
    args = parser.parse_args()
    
    try:
        # Load CSIRO projections
        projections = load_csiro_projections(args.csiro_dir, args.state)
        
        if not projections:
            raise ValueError("No CSIRO projection data found")
        
        # Generate EV demand profile
        df_demand = generate_ev_demand_profile(
            projections, args.scenario, args.since, args.until
        )
        
        # Validate data
        stats = validate_ev_data(df_demand)
        
        # Save processed data
        save_processed_data(df_demand, args.out)
        
        logger.info("EV demand ETL pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 