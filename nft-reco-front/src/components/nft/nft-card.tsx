"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { NFTItem } from "@/lib/store";
import apiService from "@/lib/api";
import useStore from "@/lib/store";

interface NFTCardProps {
  nft: NFTItem;
  showScore?: boolean;
}

export default function NFTCard({ nft, showScore = false }: NFTCardProps) {
  const { userPrefs, toggleFavorite, recordClick } = useStore();
  const [isLiked, setIsLiked] = useState(
    userPrefs.favoritedItems.includes(nft.uuid)
  );
  const [imageError, setImageError] = useState(false);

  // Placeholder image for NFTs
  // In a real implementation, you'd fetch the image from your API
  const imageUrl = `https://picsum.photos/seed/${nft.uuid}/1200/1200`;

  const handleLike = async () => {
    setIsLiked(!isLiked);
    toggleFavorite(nft.uuid);

    // Submit feedback to the API
    try {
      await apiService.submitFeedback({
        item_uuid: nft.uuid,
        user_id: userPrefs.userId,
        feedback_type: "favorite",
        value: isLiked ? 0 : 1, // 0 to remove favorite, 1 to add
      });
    } catch (error) {
      console.error("Failed to submit feedback", error);
    }
  };

  const handleClick = async () => {
    recordClick(nft.uuid);

    // Submit click feedback to the API
    try {
      await apiService.submitFeedback({
        item_uuid: nft.uuid,
        user_id: userPrefs.userId,
        feedback_type: "click",
      });
    } catch (error) {
      console.error("Failed to submit click feedback", error);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ y: -5 }}
      className="h-full"
    >
      <Card className="h-full flex flex-col overflow-hidden hover:shadow-xl transition-all duration-300">
        <CardHeader className="p-0">
          <div className="relative aspect-square overflow-hidden bg-muted">
            {!imageError ? (
              <Image
                src={imageUrl}
                alt={nft.name || "NFT image"}
                fill
                className="object-cover transition-transform hover:scale-105"
                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                onError={() => setImageError(true)}
                unoptimized
              />
            ) : (
              <div className="flex items-center justify-center w-full h-full text-muted-foreground">
                <div className="text-center p-4">
                  <p className="text-sm">Image not available</p>
                  <p className="text-xs">{nft.name}</p>
                </div>
              </div>
            )}

            {/* Hover gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
          </div>
        </CardHeader>
        <CardContent className="flex-grow p-4 sm:p-4 p-3">
          <div className="flex justify-between items-start">
            <CardTitle className="text-lg font-bold line-clamp-1">
              {nft.name || "Untitled NFT"}
            </CardTitle>
            {showScore && nft.score !== undefined && (
              <Badge variant="outline" className="ml-2 rounded-full">
                Score: {(nft.score * 100).toFixed(0)}%
              </Badge>
            )}
          </div>

          {/* Price and Collection info - Rarible style */}
          <div className="flex items-center text-sm text-muted-foreground mt-1">
            <span className="font-medium">
              {nft.categories?.[0] || "Collection"}
            </span>
          </div>

          <p className="text-sm text-muted-foreground mt-2 line-clamp-2">
            {nft.description || "No description available"}
          </p>
          <div className="mt-3 flex flex-wrap gap-1">
            {nft.tags?.slice(0, 3).map((tag) => (
              <Badge
                key={tag}
                variant="secondary"
                className="text-xs rounded-full"
              >
                {tag}
              </Badge>
            ))}
            {nft.tags && nft.tags.length > 3 && (
              <Badge variant="outline" className="text-xs rounded-full">
                +{nft.tags.length - 3}
              </Badge>
            )}
          </div>
        </CardContent>
        <CardFooter className="p-4 pt-0 flex justify-between flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleClick}
            asChild
            className="rounded-full hover:bg-primary hover:text-primary-foreground transition-colors"
          >
            <Link href={`/item/${nft.uuid}`}>View Details</Link>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleLike}
            className={`rounded-full ${isLiked ? "text-pink-500" : ""}`}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill={isLiked ? "currentColor" : "none"}
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="w-5 h-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
              />
            </svg>
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
