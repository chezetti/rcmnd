"use client";

import { useState, useEffect } from "react";
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
  });

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
        formData.append("top_k", "12");
        formData.append("diversify", "true");
        formData.append("user_id", userId);

        const response = await apiService.getRecommendations(formData);
        return response.data;
      },
      enabled: !!userId,
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
      });
    }
  }, [
    apiStats,
    userPrefs.favoritedItems.length,
    userPrefs.clickedItems.length,
  ]);

  // Создаем заглушки для любимых элементов
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
      });
    });

    return dummyItems;
  };

  // Fetch favorite items
  useEffect(() => {
    const fetchFavoriteItems = async () => {
      try {
        if (userPrefs.favoritedItems.length) {
          // В реальном приложении мы бы запрашивали детали для каждого favorites элемента по ID
          // Здесь создаем заглушки с реальными ID
          const items = createDummyFavorites(userPrefs.favoritedItems);
          setFavoriteItems(items);
        } else {
          setFavoriteItems([]);
        }
      } catch (error) {
        console.error("Failed to fetch favorite items", error);
        setFavoriteItems([]);
      }
    };

    fetchFavoriteItems();
  }, [userPrefs.favoritedItems]);

  // Set personalized items when recommendations are loaded
  useEffect(() => {
    if (recommendations?.results) {
      setPersonalizedItems(recommendations.results);
    }
  }, [recommendations]);

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
                {stats.total_clicks || userPrefs.clickedItems.length}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs for different sections */}
        <div className="mb-8">
          <div className="flex space-x-2">
            <Button
              variant={activeTab === "recommended" ? "default" : "outline"}
              onClick={() => setActiveTab("recommended")}
              className="font-medium"
              size="lg"
            >
              <Activity className="h-4 w-4 mr-2" />
              Recommended for You
            </Button>
            <Button
              variant={activeTab === "favorites" ? "default" : "outline"}
              onClick={() => setActiveTab("favorites")}
              className="font-medium"
              size="lg"
            >
              <Heart className="h-4 w-4 mr-2" />
              Your Favorites
            </Button>
          </div>
        </div>

        {activeTab === "recommended" ? (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Personalized Recommendations</h2>
            <p className="text-muted-foreground">
              Based on your browsing history and interactions.
            </p>
            <NFTGrid
              items={personalizedItems}
              isLoading={isLoadingRecommendations}
              emptyMessage="No personalized recommendations yet. Explore more NFTs to get recommendations."
              layout="masonry"
            />
          </div>
        ) : (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Your Favorites</h2>
            <p className="text-muted-foreground">
              NFTs you&apos;ve added to your favorites collection.
            </p>
            {favoriteItems.length > 0 ? (
              <NFTGrid
                items={favoriteItems}
                isLoading={false}
                emptyMessage="You haven't added any NFTs to your favorites yet."
                layout="masonry"
              />
            ) : (
              <div className="text-center py-16 border border-dashed border-border rounded-xl">
                <Heart className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-xl font-semibold mb-2">No favorites yet</h3>
                <p className="text-muted-foreground mb-6">
                  You haven&apos;t added any NFTs to your favorites collection
                  yet.
                </p>
                <Button
                  onClick={() => (window.location.href = "/")}
                  variant="outline"
                  size="lg"
                >
                  Explore NFTs
                </Button>
              </div>
            )}
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}
