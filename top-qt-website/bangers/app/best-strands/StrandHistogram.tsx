'use client';

import { useEffect, useState } from 'react';

interface HistogramData {
  generated_at: string;
  months: Array<{
    year: number;
    month: number;
    date: string;
  }>;
  strands: Record<string, {
    strand_id: string;
    total_tweets: number;
    date_range: {
      start: string;
      end: string;
    };
    histogram: number[];
  }>;
}

interface StrandHistogramProps {
  strandId: string;
  className?: string;
}

export function StrandHistogram({ strandId, className = '' }: StrandHistogramProps) {
  const [histogramData, setHistogramData] = useState<HistogramData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadHistogramData() {
      try {
        const response = await fetch('/strand_histograms.json');
        if (!response.ok) {
          throw new Error(`Failed to load histogram data: ${response.status}`);
        }
        const data: HistogramData = await response.json();
        setHistogramData(data);
      } catch (err) {
        console.error('Failed to load histogram data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load histogram data');
      } finally {
        setLoading(false);
      }
    }

    loadHistogramData();
  }, []);

  if (loading) {
    return (
      <div className={`flex items-center justify-center h-12 ${className}`}>
        <div className="text-xs text-gray-500">Loading...</div>
      </div>
    );
  }

  if (error || !histogramData) {
    return (
      <div className={`flex items-center justify-center h-12 ${className}`}>
        <div className="text-xs text-red-500">Failed to load histogram</div>
      </div>
    );
  }

  const strandData = histogramData.strands[strandId];
  if (!strandData) {
    console.warn(`No histogram data found for strand ID: ${strandId}`);
    return (
      <div className={`flex items-center justify-center h-12 ${className}`}>
        <div className="text-xs text-gray-500">No data for this strand</div>
      </div>
    );
  }

  const { histogram } = strandData;
  const maxCount = Math.max(...histogram);
  
  // Debug logging
  if (maxCount > 0) {
    console.log(`Strand ${strandId}: ${strandData.total_tweets} tweets, max: ${maxCount}, non-zero months: ${histogram.filter(c => c > 0).length}`);
  }
  
  // If no tweets, show empty histogram
  if (maxCount === 0) {
    return (
      <div className={`flex items-center justify-center h-12 ${className}`}>
        <div className="text-xs text-gray-500">No tweets found</div>
      </div>
    );
  }

  // Create histogram bars
  const containerHeight = 32; // h-8 = 32px
  const bars = histogram.map((count, index) => {
    // Calculate proportional height in pixels, scaled to container
    let height = 0;
    if (maxCount > 0 && count > 0) {
      const proportionalHeight = (count / maxCount) * containerHeight;
      // Use proportional height but ensure minimum visibility (3px minimum)
      height = Math.max(proportionalHeight, 3);
    }
    
    const monthInfo = histogramData.months[index];
    
    return (
      <div
        key={index}
        className="flex-1 flex flex-col-reverse items-center group relative min-w-0"
        title={`${monthInfo.date}: ${count} tweets`}
      >
        {/* Bar with proportional but visible heights */}
        <div
          className={`transition-opacity duration-200 group-hover:opacity-70 ${
            count > 0 ? 'bg-black' : 'bg-transparent'
          }`}
          style={{ 
            height: `${height}px`,
            width: '100%',
            minHeight: count > 0 ? `2px` : '0px' // Fallback minimum for very small container
          }}
        />
        
        {/* Tooltip on hover */}
        {count > 0 && (
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-1 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
            {monthInfo.date}: {count} tweets
          </div>
        )}
      </div>
    );
  });

  return (
    <div className={`${className}`}>
      {/* Compact header with tweet count */}
      <div className="mb-1">
        <div className="text-xs text-gray-500">
          Timeline ({strandData.total_tweets} tweets)
        </div>
      </div>
      
      {/* Compact histogram - thin but visible */}
      <div className="h-8 border border-gray-200 bg-white relative">
        <div className="h-full flex items-end">
          {bars}
        </div>
      </div>
      
      {/* Year labels positioned across the histogram */}
      <div className="relative mt-1">
        <div className="flex justify-between text-xs text-gray-400">
          <span>2015</span>
          <span>2018</span>
          <span>2021</span>
          <span>2025</span>
        </div>
      </div>
    </div>
  );
}