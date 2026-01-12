'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { fetchTweetDetails } from '@/lib/api';
import { Tweet, StrandWithTweet } from '@/lib/types';
import { TweetCard } from '../TweetCard';

interface TweetPoint {
  tweet_id: string;
  strand_id: string;
  umap_x: number;
  umap_y: number;
  has_image: boolean;
  is_root: boolean;
  media_url?: string | null;
  text_snippet: string;
  annotation: string;
  username?: string | null;
  avatar_media_url?: string | null;
  strand_title: string;
}

interface EnrichedTweetPoint extends TweetPoint {
  tweetData?: Tweet;
}

interface UmapData {
  points: TweetPoint[];
}

const COLORS_20 = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
];

interface StrandUmapProps {
  strands: StrandWithTweet[];
}

export function StrandUmap({ strands }: StrandUmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<UmapData | null>(null);
  const [enrichedData, setEnrichedData] = useState<Map<string, EnrichedTweetPoint>>(new Map());
  const [hoveredStrand, setHoveredStrand] = useState<string | null>(null);
  const [hoveredTweet, setHoveredTweet] = useState<EnrichedTweetPoint | null>(null);
  const [selectedStrand, setSelectedStrand] = useState<string | null>(null);
  const [selectedTweet, setSelectedTweet] = useState<EnrichedTweetPoint | null>(null);
  const [showConnections, setShowConnections] = useState(true);
  const [showImages, setShowImages] = useState(true);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 800 });
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number, y: number } | null>(null);
  const [isLoadingTweetData, setIsLoadingTweetData] = useState(false);
  const [connectionThreshold, setConnectionThreshold] = useState(0.1);

  // Create a map of strand_id to rating for quick lookup
  const strandRatings = useRef(new Map<string, number>());
  useEffect(() => {
    const ratingsMap = new Map<string, number>();
    strands.forEach(strand => {
      ratingsMap.set(strand.seed_tweet_id, strand.rating.rating);
    });
    strandRatings.current = ratingsMap;
  }, [strands]);

  // Load data
  useEffect(() => {
    fetch('/umap_tweets.json')
      .then(res => res.json())
      .then(setData)
      .catch(err => console.error('Failed to load UMAP data:', err));
  }, []);

  // Enrich data with real tweet details
  useEffect(() => {
    if (!data || isLoadingTweetData) return;

    const enrichTweetData = async () => {
      setIsLoadingTweetData(true);
      try {
        // Get all unique tweet IDs
        const tweetIds = [...new Set(data.points.map(p => p.tweet_id))];
        
        // Fetch tweet details in batches
        console.log(`Fetching details for ${tweetIds.length} tweets...`);
        const tweets = await fetchTweetDetails(tweetIds);
        
        // Create a map of tweet_id to Tweet
        const tweetMap = new Map<string, Tweet>();
        tweets.forEach(tweet => {
          tweetMap.set(tweet.tweet_id, tweet);
        });

        // Create enriched data map
        const enrichedMap = new Map<string, EnrichedTweetPoint>();
        data.points.forEach(point => {
          const tweetData = tweetMap.get(point.tweet_id);
          const enriched: EnrichedTweetPoint = {
            ...point,
            tweetData,
            // Override with real data if available
            username: tweetData?.username || point.username,
            avatar_media_url: tweetData?.avatar_media_url || point.avatar_media_url,
            media_url: tweetData?.media_urls?.[0] || point.media_url,
            has_image: (tweetData?.media_urls?.length || 0) > 0 || point.has_image,
            text_snippet: tweetData?.full_text || point.text_snippet
          };
          enrichedMap.set(point.tweet_id, enriched);
        });

        setEnrichedData(enrichedMap);
        console.log(`Enriched ${enrichedMap.size} tweets with real data`);

        // No image preloading needed for geometric shapes
      } catch (error) {
        console.error('Failed to enrich tweet data:', error);
      } finally {
        setIsLoadingTweetData(false);
      }
    };

    enrichTweetData();
  }, [data, isLoadingTweetData]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Color mapping
  const strandColors = useRef(new Map<string, string>());
  const getStrandColor = useCallback((strandId: string) => {
    if (!strandColors.current.has(strandId)) {
      const idx = strandColors.current.size % COLORS_20.length;
      strandColors.current.set(strandId, COLORS_20[idx]);
    }
    return strandColors.current.get(strandId)!;
  }, []);

  // Helper to draw triangle for root tweets
  const drawTriangle = useCallback((ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: string) => {
    ctx.save();
    
    // Draw white stroke first (background)
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, y - size);
    ctx.lineTo(x - size, y + size);
    ctx.lineTo(x + size, y + size);
    ctx.closePath();
    ctx.stroke();
    
    // Draw colored fill on top
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, y - size);
    ctx.lineTo(x - size, y + size);
    ctx.lineTo(x + size, y + size);
    ctx.closePath();
    ctx.fill();
    
    ctx.restore();
  }, []);

  // Draw card background helper
  const drawCardBackground = useCallback((
    ctx: CanvasRenderingContext2D, 
    x: number, 
    y: number, 
    width: number, 
    height: number, 
    color: string,
    isHovered: boolean
  ) => {
    ctx.save();
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = color;
    ctx.lineWidth = (isHovered ? 3 : 2) / Math.sqrt(transform.scale);
    ctx.shadowColor = 'rgba(0,0,0,0.1)';
    ctx.shadowBlur = 4 / Math.sqrt(transform.scale);
    ctx.shadowOffsetX = 2 / Math.sqrt(transform.scale);
    ctx.shadowOffsetY = 2 / Math.sqrt(transform.scale);
    
    ctx.fillRect(x - width/2, y - height/2, width, height);
    ctx.strokeRect(x - width/2, y - height/2, width, height);
    ctx.restore();
  }, [transform.scale]);

  // Transform coordinates from screen to world space
  const screenToWorld = useCallback((x: number, y: number) => {
    const worldX = (x / transform.scale - transform.x) / dimensions.width;
    const worldY = (y / transform.scale - transform.y) / dimensions.height;
    return { x: worldX, y: worldY };
  }, [dimensions, transform]);

  // Render canvas
  useEffect(() => {
    if (!data || !canvasRef.current || enrichedData.size === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = dimensions;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(transform.x * transform.scale, transform.y * transform.scale);
    ctx.scale(transform.scale, transform.scale);

    // Group points by strand
    const strandGroups = new Map<string, TweetPoint[]>();
    data.points.forEach(p => {
      if (!strandGroups.has(p.strand_id)) {
        strandGroups.set(p.strand_id, []);
      }
      strandGroups.get(p.strand_id)!.push(p);
    });
    
    console.log(`Drawing connections for ${strandGroups.size} strands, showConnections: ${showConnections}`);

    // Draw connections between chronologically adjacent tweets in same strand
    if (showConnections) {
      strandGroups.forEach((points, strandId) => {
        if (points.length < 2) return; // Need at least 2 points for connections

        const color = getStrandColor(strandId);
        const isStrandHovered = hoveredStrand === strandId;
        const isStrandSelected = selectedStrand === strandId;
        
        ctx.save();
        
        // Set line style based on state
        const alpha = isStrandHovered || isStrandSelected || (!hoveredStrand && !selectedStrand) ? 0.6 : 0.2;
        const lineWidth = isStrandSelected ? 2 : isStrandHovered ? 1.5 : 1;
        
        ctx.strokeStyle = color + Math.floor(alpha * 255).toString(16).padStart(2, '0');
        ctx.lineWidth = lineWidth;
        
        // Get enriched points with tweet data for sorting by date
        const enrichedPoints = points
          .map(p => ({ 
            ...p, 
            enriched: enrichedData.get(p.tweet_id) 
          }))
          .filter(p => p.enriched?.tweetData?.created_at) // Only include points with date data
          .sort((a, b) => {
            // Sort by creation date
            const dateA = new Date(a.enriched!.tweetData!.created_at).getTime();
            const dateB = new Date(b.enriched!.tweetData!.created_at).getTime();
            return dateA - dateB;
          });
        
        // Connect each tweet to its chronological neighbors if they're close enough
        for (let i = 0; i < enrichedPoints.length - 1; i++) {
          const current = enrichedPoints[i];
          const next = enrichedPoints[i + 1];
          
          // Calculate distance between chronologically adjacent tweets
          const dx = current.umap_x - next.umap_x;
          const dy = current.umap_y - next.umap_y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Draw line if chronological neighbors are close enough
          if (distance < connectionThreshold) {
            const x1 = current.umap_x * width;
            const y1 = current.umap_y * height;
            const x2 = next.umap_x * width;
            const y2 = next.umap_y * height;
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
        
        ctx.restore();
      });
    }

    // Draw points using enriched data
    data.points.forEach(point => {
      const enrichedPoint = enrichedData.get(point.tweet_id);
      if (!enrichedPoint) return;

      const x = point.umap_x * width;
      const y = point.umap_y * height;
      const color = getStrandColor(point.strand_id);
      const isStrandHovered = hoveredStrand === point.strand_id;
      const isTweetHovered = hoveredTweet?.tweet_id === point.tweet_id;
      const isStrandSelected = selectedStrand === point.strand_id;
      const isTweetSelected = selectedTweet?.tweet_id === point.tweet_id;

      ctx.save();

      // Point size and style - slightly smaller
      const baseRadius = Math.max(2, 4 / transform.scale);
      const radius = isTweetSelected ? baseRadius * 2 : isTweetHovered ? baseRadius * 1.8 : isStrandHovered || isStrandSelected ? baseRadius * 1.3 : baseRadius;
      const alpha = isStrandHovered || isStrandSelected || !hoveredStrand && !selectedStrand ? 1.0 : 0.3;
      const showFullDetail = true; // Always show full detail

      ctx.globalAlpha = alpha;

      // Root tweet - draw triangle
      if (enrichedPoint.is_root) {
        drawTriangle(ctx, x, y, radius, color);
      } else if (enrichedPoint.has_image) {
        // Tweet with image - draw square
        ctx.fillStyle = color;
        ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5 / transform.scale;
        ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
      } else {
        // Regular tweet - draw circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        if (isTweetHovered) {
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 2 / transform.scale;
          ctx.stroke();
        }
      }

      ctx.restore();
    });

    // Draw strand titles for root tweets with rating >= 7
    data.points.forEach(point => {
      const enrichedPoint = enrichedData.get(point.tweet_id);
      if (!enrichedPoint || !enrichedPoint.is_root) return;

      // Check if this strand has a rating >= 7
      const rating = strandRatings.current.get(point.strand_id);
      if (!rating || rating < 7) return;

      const x = point.umap_x * width;
      const y = point.umap_y * height;
      const color = getStrandColor(point.strand_id);
      const isStrandHovered = hoveredStrand === point.strand_id;
      const isStrandSelected = selectedStrand === point.strand_id;

      ctx.save();

      // Text styling - smaller font
      const fontSize = Math.max(8, 10 / Math.sqrt(transform.scale));
      ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      // Position text below the root triangle
      const textY = y + Math.max(2, 4 / transform.scale) + 6;
      
      // Text transparency based on strand state
      const alpha = isStrandHovered || isStrandSelected || (!hoveredStrand && !selectedStrand) ? 0.8 : 0.4;
      
      // Border transparency - more transparent by default, pop on hover/select
      const borderAlpha = isStrandHovered || isStrandSelected ? 1.0 : 0.2;
      
      // Draw text background
      const textMetrics = ctx.measureText(point.strand_title);
      const textWidth = textMetrics.width;
      const textHeight = fontSize;
      const padding = 1.5;
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(
        x - textWidth/2 - padding, 
        textY - padding, 
        textWidth + padding * 2, 
        textHeight + padding * 2
      );
      
      // Draw text border with variable transparency
      ctx.globalAlpha = borderAlpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.5;
      ctx.strokeRect(
        x - textWidth/2 - padding, 
        textY - padding, 
        textWidth + padding * 2, 
        textHeight + padding * 2
      );

      // Draw text
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.fillText(point.strand_title, x, textY);

      ctx.restore();
    });

    ctx.restore();
  }, [data, enrichedData, dimensions, hoveredStrand, hoveredTweet, selectedStrand, selectedTweet, showConnections, showImages, transform, getStrandColor, drawCardBackground, drawTriangle, connectionThreshold]);

  // Mouse event handlers
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    if (isDragging && dragStart) {
      const deltaX = clientX - dragStart.x;
      const deltaY = clientY - dragStart.y;
      setTransform(prev => ({
        ...prev,
        x: prev.x + deltaX / prev.scale,
        y: prev.y + deltaY / prev.scale
      }));
      setDragStart({ x: clientX, y: clientY });
      return;
    }

    // Transform mouse position to world coordinates
    const worldPos = screenToWorld(clientX, clientY);

    // Find nearest point
    let nearest: EnrichedTweetPoint | null = null;
    let minDist = Math.max(0.01, 0.02 / transform.scale); // threshold scales with zoom

    data.points.forEach((p: TweetPoint) => {
      const enrichedPoint = enrichedData.get(p.tweet_id);
      if (!enrichedPoint) return;

      const dx = p.umap_x - worldPos.x;
      const dy = p.umap_y - worldPos.y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < minDist) {
        minDist = dist;
        nearest = enrichedPoint;
      }
    });

    if (nearest) {
      setHoveredStrand((nearest as EnrichedTweetPoint).strand_id);
      setHoveredTweet(nearest);
    } else {
      setHoveredStrand(null);
      setHoveredTweet(null);
    }
  }, [data, enrichedData, isDragging, dragStart, transform.scale, screenToWorld]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 0) {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        setIsDragging(true);
        setDragStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      }
    }
  }, []);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current || isDragging) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;
    const worldPos = screenToWorld(clientX, clientY);

    // Find clicked tweet
    let clickedTweet: EnrichedTweetPoint | null = null;
    let minDist = Math.max(0.01, 0.02 / transform.scale);

    data.points.forEach((p: TweetPoint) => {
      const enrichedPoint = enrichedData.get(p.tweet_id);
      if (!enrichedPoint) return;

      const dx = p.umap_x - worldPos.x;
      const dy = p.umap_y - worldPos.y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      
      if (dist < minDist) {
        minDist = dist;
        clickedTweet = enrichedPoint;
      }
    });

    if (clickedTweet) {
      // Select the strand and tweet
      setSelectedStrand((clickedTweet as EnrichedTweetPoint).strand_id);
      setSelectedTweet(clickedTweet);
    } else {
      // Click away - reset selection
      setSelectedStrand(null);
      setSelectedTweet(null);
    }
  }, [data, enrichedData, isDragging, transform.scale, screenToWorld]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragStart(null);
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const worldPos = screenToWorld(mouseX, mouseY);

    const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.1, Math.min(10, transform.scale * scaleFactor));

    // Zoom towards mouse position
    const newX = mouseX / newScale - worldPos.x * dimensions.width;
    const newY = mouseY / newScale - worldPos.y * dimensions.height;

    setTransform({
      scale: newScale,
      x: newX,
      y: newY
    });
  }, [transform, dimensions, screenToWorld]);

  const handleMouseLeave = useCallback(() => {
    setHoveredStrand(null);
    setHoveredTweet(null);
    setIsDragging(false);
    setDragStart(null);
  }, []);

  // Container wheel handler to prevent page scrolling
  const handleContainerWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  // Mouse event listeners for document (for drag outside canvas)
  useEffect(() => {
    const handleDocumentMouseUp = () => {
      setIsDragging(false);
      setDragStart(null);
    };
    
    const handleDocumentMouseMove = (e: MouseEvent) => {
      if (!isDragging || !dragStart || !canvasRef.current) return;
      
      const rect = canvasRef.current.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;
      
      const deltaX = clientX - dragStart.x;
      const deltaY = clientY - dragStart.y;
      setTransform(prev => ({
        ...prev,
        x: prev.x + deltaX / prev.scale,
        y: prev.y + deltaY / prev.scale
      }));
      setDragStart({ x: clientX, y: clientY });
    };

    if (isDragging) {
      document.addEventListener('mouseup', handleDocumentMouseUp);
      document.addEventListener('mousemove', handleDocumentMouseMove);
    }

    return () => {
      document.removeEventListener('mouseup', handleDocumentMouseUp);
      document.removeEventListener('mousemove', handleDocumentMouseMove);
    };
  }, [isDragging, dragStart, transform.scale]);

  // Prevent page scrolling when mouse is over the visualization
  useEffect(() => {
    const handleDocumentWheel = (e: WheelEvent) => {
      if (containerRef.current?.contains(e.target as Node)) {
        e.preventDefault();
      }
    };

    document.addEventListener('wheel', handleDocumentWheel, { passive: false });
    
    return () => {
      document.removeEventListener('wheel', handleDocumentWheel);
    };
  }, []);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-600">Loading UMAP data...</p>
      </div>
    );
  }

  if (enrichedData.size === 0 && isLoadingTweetData) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-gray-600 mb-2">Enriching tweets with Community Archive data...</p>
          <p className="text-sm text-gray-500">This may take a moment</p>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="flex w-full h-screen bg-gray-50 overflow-hidden"
      onWheel={handleContainerWheel}
      style={{ touchAction: 'none' }}
    >
      {/* Left sidebar for tweet display */}
      <div className="w-96 flex-shrink-0 bg-white border-r-2 border-black overflow-y-auto">
        {(selectedTweet || hoveredTweet) ? (
          <div className="p-4">
            <div className="bg-white border-2 border-black shadow-[4px_4px_0_0_#000]">
              {(selectedTweet || hoveredTweet)?.tweetData ? (
                <div className="p-4">
                  <TweetCard 
                    tweet={(selectedTweet || hoveredTweet)!.tweetData!} 
                    annotation={(selectedTweet || hoveredTweet)!.annotation}
                  />
                </div>
              ) : (
                <div className="p-4">
                  <div className="flex items-start gap-3">
                    {(selectedTweet || hoveredTweet)!.avatar_media_url && (
                      <img
                        src={(selectedTweet || hoveredTweet)!.avatar_media_url!}
                        alt={(selectedTweet || hoveredTweet)!.username || ''}
                        className="w-10 h-10 rounded-full border border-gray-300 flex-shrink-0"
                      />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        {(selectedTweet || hoveredTweet)!.username && (
                          <span className="font-bold text-sm">@{(selectedTweet || hoveredTweet)!.username}</span>
                        )}
                        {(selectedTweet || hoveredTweet)!.is_root && (
                          <span className="px-2 py-0.5 text-xs font-semibold bg-yellow-100 border border-yellow-300">
                            ROOT
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-700 mb-3 leading-tight">
                        {(selectedTweet || hoveredTweet)!.text_snippet}
                      </p>
                      <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Annotation</div>
                      <p className="text-sm text-gray-700 italic leading-relaxed whitespace-pre-wrap">
                        {(selectedTweet || hoveredTweet)!.annotation}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="p-4 text-center text-gray-500">
            <p className="text-sm">Hover or click on a tweet to see details</p>
          </div>
        )}
      </div>

      {/* Main visualization area */}
      <div className="flex-1 relative">
        {/* Controls */}

        {/* Canvas */}
        <div 
          ref={containerRef} 
          className="flex items-center justify-center h-full p-4"
          onWheel={handleContainerWheel}
        >
          <canvas
            ref={canvasRef}
            className="border-2 border-black shadow-[8px_8px_0_0_#000]"
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            onWheel={handleWheel}
            onClick={handleClick}
            style={{ 
              cursor: isDragging ? 'grabbing' : hoveredTweet ? 'pointer' : 'grab',
              width: dimensions.width,
              height: dimensions.height
            }}
          />
        </div>
      </div>
    </div>
  );
}


