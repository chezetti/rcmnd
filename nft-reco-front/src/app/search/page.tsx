"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import Uppy from "@uppy/core";
import "@uppy/core/dist/style.min.css";
import "@uppy/dashboard/dist/style.min.css";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import Header from "@/components/layout/header";
import Footer from "@/components/layout/footer";
import NFTGrid from "@/components/nft/nft-grid";
import apiService from "@/lib/api";
import useStore from "@/lib/store";
import { NFTItem } from "@/lib/store";
import FilterSelect from "@/components/ui/filter-select";
import { FileUploader } from "@/components/ui/file-uploader";

export default function SearchPage() {
  const { userPrefs, setUserPrefs } = useStore();
  const [searchResults, setSearchResults] = useState<NFTItem[]>([]);
  const [description, setDescription] = useState("");
  const [balanceValue, setBalanceValue] = useState([50]); // 50% visual, 50% text
  const [searchMode, setSearchMode] = useState<
    "visual" | "textual" | "balanced"
  >((userPrefs.searchMode as "visual" | "textual" | "balanced") || "balanced");
  const uppyInstance = useRef<Uppy | null>(null);

  // Initialize Uppy
  if (!uppyInstance.current) {
    uppyInstance.current = new Uppy({
      id: "search-uppy",
      restrictions: {
        maxNumberOfFiles: 1,
        allowedFileTypes: [".jpg", ".jpeg", ".png", ".webp"],
      },
      autoProceed: false,
    });
  }

  // Update userPrefs when searchMode changes
  useEffect(() => {
    setUserPrefs({ searchMode });
  }, [searchMode, setUserPrefs]);

  // Create react-query mutation
  const searchMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await apiService.getRecommendations(formData);
      return response.data;
    },
    onSuccess: (data) => {
      setSearchResults(data.results);
    },
  });

  // Map balance value to search mode
  const getSearchModeFromBalance = (
    balance: number
  ): "visual" | "textual" | "balanced" => {
    if (balance <= 25) return "visual";
    if (balance >= 75) return "textual";
    return "balanced";
  };

  // Update balance slider when search mode changes
  useEffect(() => {
    if (searchMode === "visual") {
      setBalanceValue([0]);
    } else if (searchMode === "textual") {
      setBalanceValue([100]);
    } else {
      setBalanceValue([50]);
    }
  }, [searchMode]);

  // Handle search submission
  const handleSearch = () => {
    const formData = new FormData();

    // Add image if available
    if (uppyInstance.current && uppyInstance.current.getFiles().length > 0) {
      const file = uppyInstance.current.getFiles()[0];
      formData.append("image", file.data);
    }

    // Add other parameters
    if (description) {
      formData.append("description", description);
    }

    formData.append("top_k", "12");

    // Calculate search mode based on slider or use selected mode
    const calculatedSearchMode = getSearchModeFromBalance(balanceValue[0]);
    formData.append("search_mode", calculatedSearchMode);

    formData.append("diversify", userPrefs.diversify.toString());

    if (userPrefs.userId) {
      formData.append("user_id", userPrefs.userId);
    }

    // Execute search
    searchMutation.mutate(formData);
  };

  // Handle balance change
  const handleBalanceChange = (value: number[]) => {
    setBalanceValue(value);
    setSearchMode(getSearchModeFromBalance(value[0]));
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 container py-4 sm:py-6 md:py-10">
        <div className="mb-4 sm:mb-6 md:mb-10">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-2 sm:mb-4">
            Search NFTs
          </h1>
          <p className="text-muted-foreground">
            Find NFTs by uploading an image, describing what you&apos;re looking
            for, or both.
          </p>
        </div>

        {/* Search form с адаптивной версткой */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 md:gap-8 mb-6 sm:mb-8 md:mb-12">
          {/* Image upload */}
          <div className="p-4 sm:p-6 bg-card/30 backdrop-blur-sm rounded-lg">
            <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 flex items-center">
              <span className="bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                Upload Reference Image
              </span>
            </h2>
            <FileUploader
              uppy={uppyInstance.current}
              height={250}
              note="Only JPG, PNG and WebP images are allowed"
            />
          </div>

          {/* Right side - Description and filters с адаптивными отступами */}
          <div className="w-full">
            <div className="border rounded-lg p-4 sm:p-6 bg-card h-full">
              <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">
                Describe What You&apos;re Looking For
              </h2>
              <Textarea
                placeholder="Describe the style, theme, or specific elements you want in your NFT..."
                className="min-h-[100px] sm:min-h-[120px] mb-4 sm:mb-6"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />

              {/* Search mode select */}
              <div className="mb-4 sm:mb-6">
                <h3 className="text-sm font-medium mb-2">Search Mode</h3>
                <FilterSelect
                  placeholder="Select search mode"
                  value={searchMode}
                  onValueChange={(value) =>
                    setSearchMode(value as "visual" | "textual" | "balanced")
                  }
                  options={[
                    { value: "visual", label: "Visual" },
                    { value: "balanced", label: "Balanced" },
                    { value: "textual", label: "Textual" },
                  ]}
                />
              </div>

              {/* Search balance slider с адаптивными отступами */}
              <div className="mb-5 sm:mb-8">
                <h3 className="text-sm font-medium mb-2">Search Balance</h3>
                <div className="space-y-3 sm:space-y-4">
                  <div className="relative pt-1 pb-1">
                    {/* Сам слайдер с вариантом баланса */}
                    <Slider
                      value={balanceValue}
                      onValueChange={handleBalanceChange}
                      max={100}
                      step={1}
                      variant="balance"
                      className="z-10"
                    />
                  </div>

                  {/* Подписи слайдера с улучшенной адаптивностью */}
                  <div className="flex justify-between items-center text-xs">
                    <div className="flex items-center gap-1 sm:gap-2">
                      <div
                        className={`h-2 sm:h-3 w-2 sm:w-3 rounded-full bg-blue-500`}
                      ></div>
                      <span className={`font-medium text-blue-500`}>
                        Visual ({100 - balanceValue[0]}%)
                      </span>
                    </div>
                    <div className="flex items-center gap-1 sm:gap-2">
                      <span className={`font-medium text-purple-500`}>
                        Textual ({balanceValue[0]}%)
                      </span>
                      <div
                        className={`h-2 sm:h-3 w-2 sm:w-3 rounded-full bg-purple-500`}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Search button занимает всю ширину на мобильных */}
              <Button
                className="w-full"
                size="lg"
                onClick={handleSearch}
                disabled={searchMutation.isPending}
              >
                {searchMutation.isPending ? "Searching..." : "Search"}
              </Button>
            </div>
          </div>
        </div>

        {/* Search Results с адаптивным заголовком */}
        {searchResults.length > 0 && (
          <div>
            <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-6">
              Search Results
            </h2>
            <NFTGrid
              items={searchResults}
              layout="masonry"
              emptyMessage="No results found. Try different search parameters."
              showScores={true}
            />
          </div>
        )}

        {searchMutation.isPending && (
          <div className="text-center py-12">
            <div className="flex justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
            </div>
            <p className="text-muted-foreground mt-4">Searching for NFTs...</p>
          </div>
        )}

        {searchMutation.isError && (
          <div className="text-center py-12">
            <p className="text-red-500">
              An error occurred during search. Please try again.
            </p>
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}
