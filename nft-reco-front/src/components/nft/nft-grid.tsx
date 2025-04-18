"use client";

import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import Masonry from "react-masonry-css";
import { Loader2 } from "lucide-react";
import NFTCard from "@/components/nft/nft-card";
import { NFTItem } from "@/lib/store";
import { SkeletonCard } from "@/components/ui/skeleton-card";

interface NFTGridProps {
  items: NFTItem[];
  isLoading?: boolean;
  emptyMessage?: string;
  onLoadMore?: () => void;
  hasMore?: boolean;
  showScores?: boolean;
  layout?: "grid" | "masonry";
}

export default function NFTGrid({
  items,
  isLoading = false,
  emptyMessage = "No items found",
  onLoadMore,
  hasMore = false,
  showScores = false,
  layout = "masonry",
}: NFTGridProps) {
  // Use stable identifiers for NFTs to prevent rendering issues
  const [itemsWithIds, setItemsWithIds] = useState<
    (NFTItem & { stableId: string })[]
  >([]);
  const isLoadingMoreRef = useRef(false);
  const hasCalledInitialLoadRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState<number | null>(null);

  // Create intersection observer reference for infinite loading
  const { ref: loadMoreRef, inView } = useInView({
    threshold: 0,
    rootMargin: "0px 0px 800px 0px", // Increased margin to load earlier
  });

  // Process items to ensure they have stable IDs
  useEffect(() => {
    if (items && items.length > 0) {
      const processedItems = items.map((item, index) => ({
        ...item,
        stableId: item.uuid.startsWith("dummy-")
          ? `stable-${item.uuid}-${index}`
          : item.uuid,
      }));
      setItemsWithIds(processedItems);

      // Запоминаем текущую высоту контейнера, чтобы избежать прыжков при загрузке
      if (containerRef.current && !containerHeight) {
        setContainerHeight(containerRef.current.offsetHeight);
      }
    }
  }, [items, containerHeight]);

  // Handle load more trigger
  useEffect(() => {
    const loadMore = () => {
      if (
        inView &&
        onLoadMore &&
        hasMore &&
        !isLoading &&
        !isLoadingMoreRef.current
      ) {
        isLoadingMoreRef.current = true;
        // Debounce load more calls with setTimeout
        setTimeout(() => {
          onLoadMore();
          // Reset loading flag after a delay
          setTimeout(() => {
            isLoadingMoreRef.current = false;
          }, 1000);
        }, 100);
      }
    };

    // Инициализация загрузки только если нет элементов
    if (
      !hasCalledInitialLoadRef.current &&
      onLoadMore &&
      hasMore &&
      !isLoading &&
      items.length === 0
    ) {
      hasCalledInitialLoadRef.current = true;
      onLoadMore();
    }

    loadMore();
  }, [inView, onLoadMore, hasMore, isLoading, items.length]);

  // Masonry layout configuration
  const breakpointColumnsObj = {
    default: 4,
    1400: 4,
    1100: 3,
    768: 2,
    600: 2,
    500: 1,
  };

  // Create skeleton placeholders for loading state
  const renderSkeletons = () => {
    const count = layout === "grid" ? 8 : 12;
    return Array(count)
      .fill(0)
      .map((_, index) => (
        <div key={`skeleton-${index}`}>
          <SkeletonCard />
        </div>
      ));
  };

  // Empty state
  if (!isLoading && items.length === 0) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div
      className="mt-6 pb-6 md:pb-10"
      ref={containerRef}
      style={{
        minHeight: containerHeight ? `${containerHeight}px` : undefined,
      }}
    >
      {layout === "grid" ? (
        <div className="grid grid-cols-1 xs:grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 sm:gap-4 md:gap-6">
          {itemsWithIds.map((item) => (
            <motion.div
              key={item.stableId}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              layoutId={item.stableId}
              className="pb-2"
            >
              <NFTCard nft={item} showScore={showScores} />
            </motion.div>
          ))}

          {/* Show skeletons when loading initial content */}
          {isLoading && items.length === 0 && renderSkeletons()}
        </div>
      ) : (
        <Masonry
          breakpointCols={breakpointColumnsObj}
          className="masonry-grid"
          columnClassName="masonry-grid-column"
        >
          {itemsWithIds.map((item) => (
            <div key={item.stableId} className="masonry-item">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.3,
                  ease: "easeOut",
                }}
              >
                <NFTCard nft={item} showScore={showScores} />
              </motion.div>
            </div>
          ))}

          {/* Show skeletons when loading initial content */}
          {isLoading && items.length === 0 && renderSkeletons()}
        </Masonry>
      )}

      {/* Loading indicator and infinite scroll trigger */}
      <div
        ref={loadMoreRef}
        className="flex justify-center items-center py-4 sm:py-6 md:py-8 mt-1 sm:mt-2"
        style={{ minHeight: "60px" }}
      >
        {(isLoading || isLoadingMoreRef.current) && (
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
        )}
      </div>
    </div>
  );
}
