import { create } from "zustand";

// Define types for our NFT items
export interface NFTItem {
  uuid: string;
  score?: number;
  name?: string;
  description?: string;
  tags?: string[];
  styles?: string[];
  categories?: string[];
}

interface UserPreferences {
  userId?: string;
  favoritedItems: string[];
  clickedItems: string[];
  searchMode: "visual" | "textual" | "balanced";
  diversify: boolean;
}

interface AppState {
  // Items state
  featuredItems: NFTItem[];
  searchResults: NFTItem[];
  currentItem: NFTItem | null;
  loading: boolean;
  error: string | null;

  // User preferences
  userPrefs: UserPreferences;

  // Actions
  setFeaturedItems: (items: NFTItem[]) => void;
  setSearchResults: (items: NFTItem[]) => void;
  setCurrentItem: (item: NFTItem | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // User preference actions
  setUserId: (id: string) => void;
  toggleFavorite: (itemId: string) => void;
  recordClick: (itemId: string) => void;
  setSearchMode: (mode: "visual" | "textual" | "balanced") => void;
  setDiversify: (diversify: boolean) => void;
  setUserPrefs: (prefs: Partial<UserPreferences>) => void;
}

// Create store
const useStore = create<AppState>((set) => ({
  // Initial state
  featuredItems: [],
  searchResults: [],
  currentItem: null,
  loading: false,
  error: null,

  userPrefs: {
    userId: undefined,
    favoritedItems: [],
    clickedItems: [],
    searchMode: "balanced",
    diversify: true,
  },

  // Actions
  setFeaturedItems: (items) => set({ featuredItems: items }),
  setSearchResults: (items) => set({ searchResults: items }),
  setCurrentItem: (item) => set({ currentItem: item }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),

  // User preference actions
  setUserId: (id) =>
    set((state) => ({
      userPrefs: { ...state.userPrefs, userId: id },
    })),

  toggleFavorite: (itemId) =>
    set((state) => {
      const { favoritedItems } = state.userPrefs;
      const isFavorited = favoritedItems.includes(itemId);

      return {
        userPrefs: {
          ...state.userPrefs,
          favoritedItems: isFavorited
            ? favoritedItems.filter((id) => id !== itemId)
            : [...favoritedItems, itemId],
        },
      };
    }),

  recordClick: (itemId) =>
    set((state) => ({
      userPrefs: {
        ...state.userPrefs,
        clickedItems: [...state.userPrefs.clickedItems, itemId],
      },
    })),

  setSearchMode: (mode) =>
    set((state) => ({
      userPrefs: { ...state.userPrefs, searchMode: mode },
    })),

  setDiversify: (diversify) =>
    set((state) => ({
      userPrefs: { ...state.userPrefs, diversify },
    })),

  setUserPrefs: (prefs) =>
    set((state) => ({
      userPrefs: { ...state.userPrefs, ...prefs },
    })),
}));

export default useStore;
