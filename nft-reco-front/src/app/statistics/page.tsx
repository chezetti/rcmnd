"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Header from "@/components/layout/header";
import Footer from "@/components/layout/footer";
import NFTGrid from "@/components/nft/nft-grid";
import { Button } from "@/components/ui/button";
import { Heart, Activity, User, Share2 } from "lucide-react";
import apiService from "@/lib/api";

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

export default function StatisticsPage() {
  // Явно инициализируем с "global" для отображения Global Statistics
  const [activeTab, setActiveTab] = useState<string>("global");
  const [stats, setStats] = useState({
    total_items: 0,
    total_favorites: 0,
    total_clicks: 0,
    users_with_preferences: 0,
  });

  // Fetch global stats for the dashboard
  const { data: apiStats, isLoading: isLoadingStats } = useQuery({
    queryKey: ["global-stats"],
    queryFn: async () => {
      try {
        const response = await apiService.getAllStats();
        return response.data;
      } catch (error) {
        console.error("Failed to fetch global stats", error);
        // Возвращаем заглушки в случае ошибки
        return {
          active_items: 0,
          dimension: 0,
          index_type: "unknown",
          feedback: {
            total_clicks: 0,
            total_favorites: 0,
            users_with_preferences: 0,
          },
        };
      }
    },
    // Disable caching to always fetch fresh data on navigation
    staleTime: 0,
    refetchOnMount: "always",
  });

  // Fetch trending NFTs using the new endpoint
  const { data: trendingNFTs, isLoading: isLoadingTrending } = useQuery({
    queryKey: ["trending-nfts"],
    queryFn: async () => {
      try {
        const response = await apiService.getTrendingNFTs({ limit: 20 });
        return response.data.results || [];
      } catch (error) {
        console.error("Failed to fetch trending NFTs", error);
        return [];
      }
    },
    // Disable caching to always fetch fresh data on navigation
    staleTime: 0,
    refetchOnMount: "always",
  });

  // Update stats when API stats are loaded
  useEffect(() => {
    if (apiStats) {
      setStats({
        total_items: apiStats.active_items || 0,
        total_favorites: apiStats.feedback?.total_favorites || 0,
        total_clicks: apiStats.feedback?.total_clicks || 0,
        users_with_preferences: apiStats.feedback?.users_with_preferences || 0,
      });
    }
  }, [apiStats]);

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 container py-10">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-4">NFT Statistics</h1>
          <p className="text-muted-foreground">
            Explore NFT platform statistics and trending content.
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
                <User className="h-4 w-4 mr-2 text-primary" />
                Active Users
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoadingStats ? "..." : stats.users_with_preferences || 0}
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50 hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <Share2 className="h-4 w-4 mr-2 text-primary" />
                Total Interactions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoadingStats
                  ? "..."
                  : (stats.total_clicks || 0) + (stats.total_favorites || 0)}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs for different sections */}
        <div className="mb-8">
          <div className="flex space-x-2">
            <TabButton
              isActive={activeTab === "global"}
              onClick={() => setActiveTab("global")}
              icon={<Activity className="h-4 w-4 mr-2" />}
            >
              Global Statistics
            </TabButton>
            <TabButton
              isActive={activeTab === "trending"}
              onClick={() => setActiveTab("trending")}
              icon={<Heart className="h-4 w-4 mr-2" />}
            >
              Trending NFTs
            </TabButton>
          </div>
        </div>

        {activeTab === "global" ? (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Platform Statistics</h2>
            <p className="text-muted-foreground">
              Overview of all NFTs and user activities on the platform.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    NFT Interactions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Views/Clicks:</dt>
                      <dd className="font-medium">{stats.total_clicks || 0}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Favorites:</dt>
                      <dd className="font-medium">
                        {stats.total_favorites || 0}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    System Information
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Index Type:</dt>
                      <dd className="font-medium">
                        {apiStats?.index_type || "Unknown"}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">
                        Vector Dimension:
                      </dt>
                      <dd className="font-medium">
                        {apiStats?.dimension || 0}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Trending NFTs</h2>
            <p className="text-muted-foreground">
              Most popular NFTs based on user interactions.
            </p>
            <NFTGrid
              items={trendingNFTs || []}
              isLoading={isLoadingTrending}
              emptyMessage="No trending NFTs found at the moment."
              layout="masonry"
            />
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}
