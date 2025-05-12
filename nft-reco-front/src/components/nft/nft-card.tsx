"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { Heart } from "lucide-react";
import { motion } from "framer-motion";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { NFTItem } from "@/lib/store";
import apiService from "@/lib/api";
import useStore from "@/lib/store";
import { useToast } from "@/components/ui/use-toast";

interface NFTCardProps {
  nft: NFTItem;
}

export default function NFTCard({ nft }: NFTCardProps) {
  const { userPrefs } = useStore();
  const [imageError, setImageError] = useState(false);
  const [isLiked, setIsLiked] = useState<boolean>(nft.is_favorite || false);
  const { toast } = useToast();

  // Placeholder image for NFTs
  // In a real implementation, you'd fetch the image from your API
  const imageUrl = `https://picsum.photos/seed/${nft.uuid}/512/768`;

  const handleClick = async () => {
    // Submit click feedback to the API
    try {
      await apiService.submitFeedback({
        item_uuid: nft.uuid,
        user_id: userPrefs.userId,
        feedback_type: "click",
        value: 1.0,
      });
      console.log(`Successfully recorded click for ${nft.uuid}`);
    } catch (error) {
      console.error("Failed to submit click feedback", error);
    }
  };

  const handleLike = async () => {
    if (!userPrefs.userId) {
      toast({
        title: "Authorization required",
        description: "Please log in to add NFT to favorites",
        variant: "destructive",
      });

      return;
    }

    try {
      await apiService.submitFeedback({
        item_uuid: nft.uuid,
        user_id: userPrefs.userId,
        feedback_type: "favorite",
        value: isLiked ? 0 : 1,
      });
      // Update local state after successful API call
      setIsLiked(!isLiked);
    } catch (error) {
      console.error("Failed to submit like feedback", error);
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
      <Card className="overflow-hidden h-full bg-card border-border">
        <div className="p-0 relative">
          <Link href={`/item/${nft.uuid}`} onClick={handleClick}>
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
            </div>
          </Link>

          {/* Показатель соответствия алгоритма в левом верхнем углу */}
          {nft.score !== undefined && (
            <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm rounded-md px-2 py-1">
              <div className="text-xs font-medium text-[#F4F5F6] flex items-center">
                <span>Score: {(nft.score * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}
        </div>

        <CardContent className="p-4">
          <div className="flex justify-between items-center">
            <Link
              href={`/item/${nft.uuid}`}
              onClick={handleClick}
              className="hover:text-primary transition-colors"
            >
              <h3 className="text-base font-bold line-clamp-1">
                {nft.name || "Untitled NFT"}
              </h3>
            </Link>

            {/* Иконка избранного выровнена с заголовком */}
            <button
              className={`transition-colors ${
                isLiked
                  ? "text-red-500"
                  : "text-muted-foreground hover:text-red-500"
              }`}
              onClick={handleLike}
            >
              <Heart
                className="h-5 w-5"
                fill={isLiked ? "currentColor" : "none"}
              />
            </button>
          </div>

          {/* Описание коллекции */}
          <div className="flex items-center text-xs text-muted-foreground mt-1">
            <span className="font-medium">{nft.description}</span>
          </div>

          {/* Блок со стилями */}
          <div className="flex flex-wrap gap-6 mt-3 text-xs">
            {nft.styles?.slice(0, 2).map((style) => (
              <div key={style}>
                <span className="text-muted-foreground">{style}</span>
              </div>
            ))}
            {nft.styles && nft.styles.length > 2 && (
              <Badge variant="outline" className="text-xs rounded-full">
                +{nft.styles.length - 2}
              </Badge>
            )}
          </div>

          {/* Дополнительные теги как на примере */}
          <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground">
            {nft.tags?.slice(0, 2).map((tag) => (
              <Badge
                key={tag}
                variant="secondary"
                className="text-xs px-2 py-0"
              >
                {tag}
              </Badge>
            ))}
            {nft.tags && nft.tags.length > 2 && (
              <Badge variant="outline" className="text-xs rounded-full">
                +{nft.tags.length - 2}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
