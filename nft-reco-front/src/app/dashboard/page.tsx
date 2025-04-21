"use client";

import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Header from "@/components/layout/header";
import Footer from "@/components/layout/footer";
import NFTGrid from "@/components/nft/nft-grid";
import { Button } from "@/components/ui/button";
import { Heart, Clock, Activity } from "lucide-react";
import apiService from "@/lib/api";
import useStore from "@/lib/store";
import { NFTItem } from "@/lib/store";

// Компонент унифицированной кнопки таба для многократного использования
interface TabButtonProps {
  isActive: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function TabButton({ isActive, onClick, icon, children }: TabButtonProps) {
  return (
    <Button
      variant={isActive ? "default" : "outline"}
      onClick={onClick}
      className={`font-medium ${isActive ? "border-2 border-primary" : ""}`}
      size="lg"
    >
      {icon}
      {children}
    </Button>
  );
}

export default function DashboardPage() {
  const { userPrefs } = useStore();
  const [personalizedItems, setPersonalizedItems] = useState<NFTItem[]>([]);
  const [favoriteItems, setFavoriteItems] = useState<NFTItem[]>([]);
  const [activeTab, setActiveTab] = useState<string>("recommended");
  const [stats, setStats] = useState({
    total_items: 0,
    total_favorites: 0,
    total_clicks: 0,
    total_purchases: 0,
    users_with_preferences: 0,
    total_interactions: 0,
  });
  const [favoritesParams, setFavoritesParams] = useState({
    limit: 20,
    offset: 0,
  });
  const [hasMoreFavorites, setHasMoreFavorites] = useState(true);

  // Get user ID, if not available use a fallback
  const userId = userPrefs.userId || "guest-user";

  // Fetch personalized recommendations
  const { data: recommendations, isLoading: isLoadingRecommendations } =
    useQuery({
      queryKey: ["personalized-recommendations", userId],
      queryFn: async () => {
        // Create form data for personalized recommendations
        const formData = new FormData();
        formData.append("description", "Personalized recommendations");
        formData.append("top_k", "15");
        formData.append("diversify", "true");
        formData.append("user_id", userId);

        const response = await apiService.getRecommendations(formData);
        return response.data;
      },
      enabled: !!userId,
      // Disable caching to always fetch fresh data on navigation
      staleTime: 0,
      refetchOnMount: "always",
    });

  // Fetch stats for the dashboard
  const { data: apiStats, isLoading: isLoadingStats } = useQuery({
    queryKey: ["stats"],
    queryFn: async () => {
      try {
        const response = await apiService.getStats();
        return response.data;
      } catch (error) {
        console.error("Failed to fetch stats", error);
        // Возвращаем заглушки в случае ошибки
        return {
          active_items: 1600,
          dimension: 1280,
          index_type: "flat",
          feedback: {
            total_clicks: 8,
            total_favorites: 8,
            total_purchases: 0,
            users_with_preferences: 0,
          },
        };
      }
    },
    // Disable caching to always fetch fresh data
    staleTime: 0,
    refetchOnMount: true,
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch favorite items with pagination from backend API
  const { data: favItems, isLoading: isLoadingFavorites } = useQuery({
    queryKey: ["favorites", favoritesParams, userPrefs.userId],
    queryFn: async () => {
      try {
        // Only fetch from API if user is logged in (has userId)
        if (userPrefs.userId) {
          // Fetch favorites directly from the backend API
          const response = await apiService.getUserFavorites(favoritesParams);
          return response.data;
        }
        // Fallback to local favorites when not logged in
        else if (userPrefs.favoritedItems.length > 0) {
          // Calculate items to fetch based on pagination
          const { offset, limit } = favoritesParams;

          // Get subset of favorited items for current page
          const itemsToFetch = userPrefs.favoritedItems.slice(
            offset,
            offset + limit
          );

          if (itemsToFetch.length === 0) {
            return { results: [], total: userPrefs.favoritedItems.length };
          }

          // Create placeholder items with real IDs as a fallback
          const items = createDummyFavorites(itemsToFetch);

          return {
            results: items,
            total: userPrefs.favoritedItems.length,
          };
        }
        return { results: [], total: 0 };
      } catch (error) {
        console.error("Failed to fetch favorite items", error);

        // If the API fails, fallback to locally stored favorites
        if (userPrefs.favoritedItems.length > 0) {
          // Calculate items to fetch based on pagination
          const { offset, limit } = favoritesParams;

          // Get subset of favorited items for current page
          const itemsToFetch = userPrefs.favoritedItems.slice(
            offset,
            offset + limit
          );

          if (itemsToFetch.length === 0) {
            return { results: [], total: userPrefs.favoritedItems.length };
          }

          // Create placeholder items with real IDs as a fallback
          const items = createDummyFavorites(itemsToFetch);

          return {
            results: items,
            total: userPrefs.favoritedItems.length,
          };
        }
        return { results: [], total: 0 };
      }
    },
    enabled: activeTab === "favorites",
    // Disable caching to always fetch fresh data on navigation
    staleTime: 0,
    refetchOnMount: "always",
  });

  // Update stats when API stats are loaded
  useEffect(() => {
    if (apiStats) {
      setStats({
        total_items: apiStats.active_items || 0,
        total_favorites:
          apiStats.feedback?.total_favorites || userPrefs.favoritedItems.length,
        total_clicks:
          apiStats.feedback?.total_clicks || userPrefs.clickedItems.length,
        total_purchases: apiStats.feedback?.total_purchases || 0,
        users_with_preferences: apiStats.feedback?.users_with_preferences || 0,
        total_interactions:
          apiStats.feedback?.total_interactions ||
          userPrefs.clickedItems.length + userPrefs.favoritedItems.length,
      });
    }
  }, [
    apiStats,
    userPrefs.favoritedItems.length,
    userPrefs.clickedItems.length,
  ]);

  // Create fallback dummy items for favorites (used only if API fails)
  const createDummyFavorites = (ids: string[]) => {
    const dummyItems: NFTItem[] = [];
    const categories = ["art", "collectible", "game", "metaverse", "defi"];
    const styles = ["pixel", "3d", "abstract", "realistic", "surreal"];

    ids.forEach((id, index) => {
      const category =
        categories[Math.floor(Math.random() * categories.length)];
      const style = styles[Math.floor(Math.random() * styles.length)];
      const tags = [`favorite`, `nft`, `${category}`, `${style}`];

      dummyItems.push({
        uuid: id,
        name: `Favorite NFT #${index + 1}`,
        description: `Your favorite NFT with ${category} category and ${style} style.`,
        categories: [category],
        styles: [style],
        tags: tags,
        creator: "Unknown Artist",
        image_url: `https://api.dicebear.com/6.x/shapes/svg?seed=${id}`,
        category: category,
        currency: "ETH",
        price: 0.1 + Math.random() * 2,
      });
    });

    return dummyItems;
  };

  // Set personalized items when recommendations are loaded
  useEffect(() => {
    if (recommendations?.results) {
      setPersonalizedItems(recommendations.results);
    }
  }, [recommendations]);

  // Update favorite items when data is loaded
  useEffect(() => {
    if (favItems?.results) {
      if (favoritesParams.offset === 0) {
        // First page - replace all items
        setFavoriteItems(favItems.results);
      } else {
        // Subsequent pages - append new items
        setFavoriteItems((prev) => {
          const newItems = [...prev];
          favItems.results.forEach((item: NFTItem) => {
            if (!newItems.some((existing) => existing.uuid === item.uuid)) {
              newItems.push(item);
            }
          });
          return newItems;
        });
      }

      // Update hasMoreFavorites status
      setHasMoreFavorites(
        favoritesParams.offset + favoritesParams.limit < (favItems.total || 0)
      );
    }
  }, [favItems, favoritesParams.offset, favoritesParams.limit]);

  // Handle load more favorites
  const handleLoadMoreFavorites = useCallback(() => {
    if (!isLoadingFavorites && hasMoreFavorites) {
      setFavoritesParams((prev) => ({
        ...prev,
        offset: prev.offset + prev.limit,
      }));
    }
  }, [isLoadingFavorites, hasMoreFavorites]);

  // Reset pagination when switching to favorites tab
  useEffect(() => {
    if (activeTab === "favorites") {
      setFavoritesParams({
        limit: 20,
        offset: 0,
      });
    }
  }, [activeTab]);

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 container py-10">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Your Dashboard</h1>
          <p className="text-muted-foreground">
            View your personalized recommendations and favorite NFTs.
          </p>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <Card className="border-border/50 hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <Activity className="h-4 w-4 mr-2 text-primary" />
                Total NFTs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoadingStats ? "..." : stats.total_items || 0}
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50 hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <Heart className="h-4 w-4 mr-2 text-primary" />
                Your Favorites
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {stats.total_favorites || userPrefs.favoritedItems.length}
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50 hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <Clock className="h-4 w-4 mr-2 text-primary" />
                Your Interactions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoadingStats ? (
                  <span className="text-muted animate-pulse">...</span>
                ) : (
                  stats.total_interactions || userPrefs.clickedItems.length
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs for different sections */}
        <div className="mb-8">
          <div className="flex space-x-2">
            <TabButton
              isActive={activeTab === "recommended"}
              onClick={() => setActiveTab("recommended")}
              icon={<Activity className="h-4 w-4 mr-2" />}
            >
              Recommended for You
            </TabButton>
            <TabButton
              isActive={activeTab === "favorites"}
              onClick={() => setActiveTab("favorites")}
              icon={<Heart className="h-4 w-4 mr-2" />}
            >
              Your Favorites
            </TabButton>
          </div>
        </div>

        {activeTab === "recommended" ? (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Personalized Recommendations</h2>
            <p className="text-muted-foreground">
              Discover NFTs tailored to your preferences.
            </p>
            <NFTGrid
              items={personalizedItems}
              isLoading={isLoadingRecommendations}
              emptyMessage="No personalized recommendations found. Try exploring more NFTs to improve your recommendations."
              layout="masonry"
            />
          </div>
        ) : (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Your Favorites</h2>
            <p className="text-muted-foreground">
              NFTs you&apos;ve added to your favorites collection.
            </p>
            {favoriteItems.length === 0 && !isLoadingFavorites ? (
              <div className="flex flex-col items-center justify-center py-20">
                <Heart className="h-16 w-16 text-muted-foreground mb-4" />
                <h2 className="text-2xl font-semibold mb-2">
                  No favorites yet
                </h2>
                <p className="text-muted-foreground mb-6">
                  You haven&apos;t added any NFTs to your favorites collection
                  yet.
                </p>
              </div>
            ) : (
              <NFTGrid
                items={favoriteItems}
                isLoading={isLoadingFavorites}
                emptyMessage="No favorites found."
                onLoadMore={handleLoadMoreFavorites}
                hasMore={hasMoreFavorites}
                layout="masonry"
              />
            )}
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}
