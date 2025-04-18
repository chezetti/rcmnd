"use client";

import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import Masonry from "react-masonry-css";
import { Loader2 } from "lucide-react";
import NFTCard from "@/components/nft/nft-card";
import { NFTItem } from "@/lib/store";
import { Skeleton } from "@/components/ui/skeleton";

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
    default: 5, // More columns for larger screens
    1600: 5,
    1300: 4,
    1000: 3,
    768: 2,
    500: 1,
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
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-3 w-full">
          {isLoading
            ? Array.from({ length: 8 }).map((_, i) => (
                <motion.div
                  key={`skeleton-${i}`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="w-full"
                >
                  <Skeleton className="w-full h-[280px] rounded-lg" />
                </motion.div>
              ))
            : itemsWithIds.map((item) => (
                <motion.div
                  key={item.stableId}
                  layout
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <NFTCard nft={item} />
                </motion.div>
              ))}
        </div>
      ) : (
        <Masonry
          breakpointCols={breakpointColumnsObj}
          className="flex w-full"
          columnClassName="bg-transparent pl-0 pr-3 first:pl-0 last:pr-0"
        >
          {isLoading
            ? Array.from({ length: 8 }).map((_, i) => (
                <motion.div
                  key={`skeleton-${i}`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="mb-3"
                >
                  <Skeleton className="w-full h-[280px] rounded-lg" />
                </motion.div>
              ))
            : itemsWithIds.map((item) => (
                <motion.div
                  key={item.stableId}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="mb-3"
                >
                  <NFTCard nft={item} />
                </motion.div>
              ))}
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
