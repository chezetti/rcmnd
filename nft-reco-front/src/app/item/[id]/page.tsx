"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import Header from "@/components/layout/header";
import Footer from "@/components/layout/footer";
import NFTGrid from "@/components/nft/nft-grid";
import apiService from "@/lib/api";
import useStore from "@/lib/store";
import { NFTItem } from "@/lib/store";
import { useParams } from "next/navigation";
import { useToast } from "@/components/ui/use-toast";

export default function ItemPage() {
  const router = useRouter();
  const params = useParams();
  const itemId =
    typeof params.id === "string"
      ? params.id
      : Array.isArray(params.id)
      ? params.id[0]
      : "";

  const { userPrefs } = useStore();
  const [isLiked, setIsLiked] = useState(false);
  const [similarItems, setSimilarItems] = useState<NFTItem[]>([]);
  const { toast } = useToast();

  // Fetch NFT details
  const {
    data: item,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["item", itemId],
    queryFn: async () => {
      const response = await apiService.getItem(itemId);
      return response.data;
    },
    enabled: !!itemId,
  });

  // Fetch similar items
  const { data: recommendations, isLoading: isLoadingSimilar } = useQuery({
    queryKey: ["recommendations", itemId],
    queryFn: async () => {
      // Create form data with the item ID for recommendation
      const formData = new FormData();
      formData.append(
        "description",
        item?.description || `Similar to item ${itemId}`
      );
      formData.append("top_k", "6");
      formData.append("diversify", "true");
      if (userPrefs.userId) {
        formData.append("user_id", userPrefs.userId);
      }

      const response = await apiService.getRecommendations(formData);
      return response.data;
    },
    enabled: !!item && !!itemId, // Only fetch when item is loaded
  });

  // Update like status based on user favorites from backend
  useEffect(() => {
    setIsLiked(item?.is_favorite || false);
  }, [item?.is_favorite]);

  // Set similar items when recommendations are loaded
  useEffect(() => {
    if (recommendations?.results) {
      // Filter out the current item from results if present
      setSimilarItems(
        recommendations.results.filter((rec: NFTItem) => rec.uuid !== itemId)
      );
    }
  }, [recommendations, itemId]);

  // Handle like button click
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
        item_uuid: itemId,
        user_id: userPrefs.userId,
        feedback_type: "favorite",
        value: isLiked ? 0 : 1,
      });
      // After successful API call, update local state
      setIsLiked(!isLiked);
    } catch (error) {
      console.error("Failed to submit feedback", error);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className="flex-1 container-fluid py-10 px-4 md:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Skeleton className="aspect-square rounded-lg" />
            <div className="space-y-4">
              <Skeleton className="h-8 w-3/4" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
              <div className="flex gap-2 mt-4">
                <Skeleton className="h-8 w-20" />
                <Skeleton className="h-8 w-20" />
              </div>
            </div>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  // Error state
  if (isError || !item) {
    return (
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className="flex-1 container-fluid py-10 px-4 md:px-8 text-center">
          <h2 className="text-2xl font-bold mb-4">Item Not Found</h2>
          <p className="text-muted-foreground mb-8">
            The NFT you&apos;re looking for doesn&apos;t exist or has been
            removed.
          </p>
          <Button onClick={() => router.push("/")}>Back to Explore</Button>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 container max-w-6xl mx-auto py-4 sm:py-6 md:py-10 px-4 md:px-8">
        {/* Main content area with centered layout */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr,1fr] gap-8 mb-8 md:mb-16">
          {/* Left side - Image */}
          <div className="flex justify-center items-start">
            <Card className="overflow-hidden rounded-lg border w-full max-w-md">
              <CardContent className="p-0">
                <div className="aspect-square relative">
                  <Image
                    src={`https://picsum.photos/seed/${itemId}/512/512`}
                    alt={item.name || "NFT Image"}
                    fill
                    className="object-cover"
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                    priority
                    unoptimized
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right side - NFT details */}
          <div className="space-y-4 sm:space-y-6">
            <div className="flex flex-col">
              <div className="flex justify-between items-start flex-wrap gap-2">
                <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold break-words">
                  {item.name || "Untitled NFT"}
                </h1>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleLike}
                  className={`rounded-full ${isLiked ? "text-red-500" : ""}`}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill={isLiked ? "currentColor" : "none"}
                    viewBox="0 0 24 24"
                    strokeWidth={1.5}
                    stroke="currentColor"
                    className="w-6 h-6"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
                    />
                  </svg>
                </Button>
              </div>
              <div className="mt-1 flex items-center text-sm text-muted-foreground">
                <span className="font-medium">
                  {item.categories?.[0] || "Collection"}
                </span>
                {item.styles && item.styles.length > 0 && (
                  <span className="mx-2">•</span>
                )}
                <span>{item.styles?.[0] || ""}</span>
              </div>
            </div>

            <div className="pt-4">
              <h2 className="text-lg font-semibold mb-2">Description</h2>
              <p className="text-muted-foreground">
                {item.description || "No description available"}
              </p>
            </div>

            {item.tags && item.tags.length > 0 && (
              <div className="pt-4">
                <h2 className="text-lg font-semibold mb-2">Tags</h2>
                <div className="flex flex-wrap gap-2">
                  {item.tags.map((tag: string) => (
                    <Badge key={tag} variant="secondary">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Similar Items section */}
        <div className="mt-6 sm:mt-8 md:mt-12">
          <h2 className="text-xl sm:text-2xl font-bold mb-4">Similar Items</h2>
          {isLoadingSimilar ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {Array(5)
                .fill(0)
                .map((_, index) => (
                  <div key={index} className="aspect-square">
                    <Skeleton className="w-full h-full rounded-lg" />
                  </div>
                ))}
            </div>
          ) : similarItems.length > 0 ? (
            <NFTGrid items={similarItems} />
          ) : (
            <p className="text-muted-foreground">No similar items found</p>
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
}
