"use client";

import { useEffect, useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import Header from "@/components/layout/header";
import Footer from "@/components/layout/footer";
import NFTGrid from "@/components/nft/nft-grid";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, SlidersHorizontal, Filter } from "lucide-react";
import apiService from "@/lib/api";
import useStore from "@/lib/store";
import { NFTItem } from "@/lib/store";
import FilterSelect from "@/components/ui/filter-select";

export default function ExplorePage() {
  const { setFeaturedItems } = useStore();
  const [searchParams, setSearchParams] = useState({
    category: "",
    style: "",
    tags: "",
    limit: 20,
    offset: 0,
    sort_by: "popularity",
  });
  const [allItems, setAllItems] = useState<NFTItem[]>([]);
  const [hasMore, setHasMore] = useState(true);
  const [showMoreFilters, setShowMoreFilters] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Создаем заглушки для отображения в случае, если API не возвращает данные
  const createDummyItems = () => {
    const dummyItems: NFTItem[] = [];
    const categories = ["art", "collectible", "game", "metaverse", "defi"];
    const styles = ["pixel", "3d", "abstract", "realistic", "surreal"];

    for (let i = 1; i <= 12; i++) {
      const category =
        categories[Math.floor(Math.random() * categories.length)];
      const style = styles[Math.floor(Math.random() * styles.length)];
      const tags = [`tag${i}`, `nft`, `${category}`, `${style}`];

      dummyItems.push({
        uuid: `dummy-${i}-${Date.now()}-${Math.random()
          .toString(36)
          .substring(2, 9)}`,
        name: `NFT #${i}`,
        description: `This is a placeholder NFT with ${category} category and ${style} style.`,
        categories: [category],
        styles: [style],
        tags: tags,
      });
    }

    return dummyItems;
  };

  // Fetch NFTs with filters and pagination
  const { data, isLoading } = useQuery({
    queryKey: ["explore", searchParams],
    queryFn: async () => {
      try {
        console.log("Fetching NFTs with params:", searchParams);
        const response = await apiService.explore(searchParams);

        // Если API вернуло пустые результаты, используем заглушки
        if (!response.data?.results || response.data.results.length === 0) {
          const dummyData = {
            results: createDummyItems(),
            total: 50,
          };
          return dummyData;
        }
        return response.data || { results: [], total: 0 };
      } catch (error) {
        console.error("Error fetching NFTs:", error);
        // В случае ошибки также возвращаем заглушки
        return { results: createDummyItems(), total: 50 };
      }
    },
    placeholderData: (previousData) => previousData,
    // Отключаем автоматическую загрузку при монтировании компонента
    // Это позволит избежать множественных запросов при инициализации
    enabled: Boolean(searchParams),
    // Кэшируем результаты на 5 минут, чтобы избежать лишних запросов
    staleTime: 5 * 60 * 1000,
    // Предотвращаем повторные запросы при потере фокуса и повторном подключении
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    // Отключаем повторные попытки при ошибке
    retry: false,
  });

  // Безопасный доступ к данным
  const results = data?.results || [];
  const total = data?.total || 0;

  // Update store with featured items on initial load
  useEffect(() => {
    if (results.length > 0 && searchParams.offset === 0) {
      // Для начальной загрузки - заменяем все элементы
      setFeaturedItems(results);
      setAllItems(results);
      setHasMore(total > results.length);
    } else if (results.length > 0 && searchParams.offset > 0) {
      // Для последующих загрузок - добавляем новые элементы
      setAllItems((prev) => {
        // Create a unique list by combining old and new items, removing duplicates
        const newItems = [...prev];
        results.forEach((item: NFTItem) => {
          if (!newItems.some((existing) => existing.uuid === item.uuid)) {
            newItems.push(item);
          }
        });
        return newItems;
      });

      // Update hasMore status более надежным образом
      setHasMore(searchParams.offset + results.length < total);
    }
  }, [results, total, searchParams.offset, setFeaturedItems]);

  // Handle filter changes
  const handleFilterChange = useCallback((key: string, value: string) => {
    // Если выбрано "all", то устанавливаем пустую строку для совместимости с API
    const apiValue = value === "all" ? "" : value;

    setSearchParams((prev) => ({
      ...prev,
      [key]: apiValue,
      offset: 0, // Reset pagination on filter change
    }));
    setAllItems([]);
  }, []);

  // Load more items
  const handleLoadMore = useCallback(() => {
    if (!isLoading && hasMore) {
      setSearchParams((prev) => ({
        ...prev,
        offset: prev.offset + prev.limit,
      }));
    }
  }, [isLoading, hasMore]);

  // Сброс всех фильтров
  const resetFilters = useCallback(() => {
    setSearchParams({
      category: "",
      style: "",
      tags: "",
      limit: 20,
      offset: 0,
      sort_by: "popularity",
    });
    setAllItems([]);
  }, []);

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 container-fluid py-10 px-4 md:px-8">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Explore NFTs</h1>
          <p className="text-muted-foreground">
            Discover unique NFTs from various collections, curated for you.
          </p>
        </div>

        {/* Filters toggle button */}
        <div className="flex justify-end mb-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2"
          >
            <Filter className="h-4 w-4" />
            {showFilters ? "Hide Filters" : "Filters"}
          </Button>
        </div>

        {/* Collapsible Filters */}
        {showFilters && (
          <div className="border rounded-lg p-4 md:p-6 bg-card mb-8 animate-in fade-in-50 slide-in-from-top-5 duration-300">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Filters</h2>
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowMoreFilters(!showMoreFilters)}
                >
                  <SlidersHorizontal className="h-4 w-4 mr-2" />
                  {showMoreFilters ? "Less Filters" : "More Filters"}
                </Button>
                <Button variant="outline" size="sm" onClick={resetFilters}>
                  Reset Filters
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div>
                <FilterSelect
                  placeholder="Category"
                  value={searchParams.category || "all"}
                  onValueChange={(value) =>
                    handleFilterChange("category", value)
                  }
                  options={[
                    { value: "all", label: "All Categories" },
                    { value: "art", label: "Art" },
                    { value: "collectible", label: "Collectible" },
                    { value: "game", label: "Game" },
                    { value: "metaverse", label: "Metaverse" },
                    { value: "defi", label: "DeFi" },
                  ]}
                />
              </div>
              <div>
                <FilterSelect
                  placeholder="Style"
                  value={searchParams.style || "all"}
                  onValueChange={(value) => handleFilterChange("style", value)}
                  options={[
                    { value: "all", label: "All Styles" },
                    { value: "pixel", label: "Pixel" },
                    { value: "3d", label: "3D" },
                    { value: "abstract", label: "Abstract" },
                    { value: "realistic", label: "Realistic" },
                    { value: "surreal", label: "Surreal" },
                  ]}
                />
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="Search by tags (comma separated)"
                  value={searchParams.tags}
                  onChange={(e) => handleFilterChange("tags", e.target.value)}
                />
                <Button variant="ghost" size="icon">
                  <Search className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {showMoreFilters && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                <div>
                  <FilterSelect
                    placeholder="Sort by"
                    value={searchParams.sort_by}
                    onValueChange={(value) =>
                      handleFilterChange("sort_by", value)
                    }
                    options={[
                      { value: "popularity", label: "Popularity" },
                      { value: "latest", label: "Latest" },
                      { value: "price_low", label: "Price: Low to High" },
                      { value: "price_high", label: "Price: High to Low" },
                    ]}
                  />
                </div>
                <div>
                  <FilterSelect
                    placeholder="Items per page"
                    value={searchParams.limit.toString()}
                    onValueChange={(value) =>
                      handleFilterChange("limit", value)
                    }
                    options={[
                      { value: "10", label: "10 per page" },
                      { value: "20", label: "20 per page" },
                      { value: "50", label: "50 per page" },
                    ]}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Featured/Results display */}
        <NFTGrid
          items={allItems}
          isLoading={isLoading && allItems.length === 0}
          onLoadMore={handleLoadMore}
          hasMore={hasMore}
          emptyMessage="No NFTs found. Try different filters."
        />
      </main>
      <Footer />
    </div>
  );
}
