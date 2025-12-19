#!/usr/bin/env python3
"""
Generate histogram data for each strand from tweet_embeddings_atlas.parquet
Creates monthly tweet count histograms from January 2010 to December 2025
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys

def generate_strand_histograms():
    # Paths
    parquet_path = Path("../../scratchpads/data/tweet_embeddings_atlas.parquet")
    output_path = Path("./public/strand_histograms.json")
    
    print(f"Loading parquet file: {parquet_path}")
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        sys.exit(1)
    
    # Load the data
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} tweets from {len(df['strand_id'].unique())} strands")
    
    # Convert dates to datetime
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['datetime'])  # Remove tweets without dates
    print(f"After filtering out missing dates: {len(df)} tweets")
    
    # Create monthly bins from Jan 2015 to Dec 2025
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # Generate all months in the range
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
    months = [{"year": d.year, "month": d.month, "date": d.strftime("%Y-%m")} for d in date_range]
    print(f"Created {len(months)} monthly bins from {months[0]['date']} to {months[-1]['date']}")
    
    # Group by strand and calculate histograms
    strand_histograms = {}
    
    for strand_id in df['strand_id'].unique():
        strand_tweets = df[df['strand_id'] == strand_id].copy()
        
        # Add year-month column
        strand_tweets['year_month'] = strand_tweets['datetime'].dt.to_period('M')
        
        # Count tweets per month
        monthly_counts = strand_tweets.groupby('year_month').size()
        
        # Create histogram array matching our date range
        histogram = []
        for month_info in months:
            period_key = pd.Period(f"{month_info['year']}-{month_info['month']:02d}", freq='M')
            count = monthly_counts.get(period_key, 0)
            histogram.append(int(count))  # Convert to regular int for JSON serialization
        
        strand_histograms[strand_id] = {
            "strand_id": strand_id,
            "total_tweets": int(len(strand_tweets)),
            "date_range": {
                "start": months[0]['date'],
                "end": months[-1]['date']
            },
            "histogram": histogram
        }
        
        print(f"Strand {strand_id}: {len(strand_tweets)} tweets, max monthly count: {max(histogram) if histogram else 0}")
    
    # Save to JSON
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "months": months,
        "strands": strand_histograms
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved histogram data to {output_path}")
    print(f"Generated histograms for {len(strand_histograms)} strands")
    
    # Show some statistics
    total_tweet_counts = [data['total_tweets'] for data in strand_histograms.values()]
    print(f"Tweet counts per strand: min={min(total_tweet_counts)}, max={max(total_tweet_counts)}, avg={sum(total_tweet_counts)/len(total_tweet_counts):.1f}")

if __name__ == "__main__":
    generate_strand_histograms()