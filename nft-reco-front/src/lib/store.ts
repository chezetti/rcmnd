import { create } from "zustand";

// Define types for our NFT items
export interface NFTItem {
  uuid: string;
  name: string;
  description?: string;
  creator: string;
  image_url: string;
  category: string;
  style?: string;
  tags?: string[];
  price: number;
  currency: string;
  created_at?: string;
  updated_at?: string;
  score?: number;
  styles?: string[];
  categories?: string[];
  is_favorite?: boolean;
}

interface UserPreferences {
  userId?: string;
  favoritedItems: string[];
  clickedItems: string[];
  searchMode: "visual" | "textual" | "balanced";
  diversify: boolean;
}

export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  role: string;
  created_at: string;
}

interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  loading: boolean;
  error: string | null;
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

  // Authentication state
  auth: AuthState;

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

  // Authentication actions
  loginSuccess: (user: User, token: string) => void;
  logout: () => void;
  setAuthLoading: (loading: boolean) => void;
  setAuthError: (error: string | null) => void;
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

  // Authentication state
  auth: {
    isAuthenticated: false,
    user: null,
    loading: false,
    error: null,
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
      const newFavoritedItems = isFavorited
        ? favoritedItems.filter((id) => id !== itemId)
        : [...favoritedItems, itemId];

      return {
        userPrefs: {
          ...state.userPrefs,
          favoritedItems: newFavoritedItems,
        },
      };
    }),

  recordClick: (itemId) =>
    set((state) => {
      // Only add the item if it's not already in the clicked items list
      if (!state.userPrefs.clickedItems.includes(itemId)) {
        return {
          userPrefs: {
            ...state.userPrefs,
            clickedItems: [...state.userPrefs.clickedItems, itemId],
          },
        };
      }
      return state; // Return unchanged state if item was already clicked
    }),

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

  // Authentication actions
  loginSuccess: (user, token) => {
    // Save token to localStorage
    if (typeof window !== "undefined") {
      localStorage.setItem("auth_token", token);
    }
    // Update state
    set((state) => ({
      auth: {
        ...state.auth,
        isAuthenticated: true,
        user,
        loading: false,
        error: null,
      },
      userPrefs: {
        ...state.userPrefs,
        userId: user.id,
      },
    }));
  },

  logout: () => {
    // Remove token from localStorage
    if (typeof window !== "undefined") {
      localStorage.removeItem("auth_token");
    }
    // Reset auth state
    set((state) => ({
      auth: {
        ...state.auth,
        isAuthenticated: false,
        user: null,
        error: null,
      },
      userPrefs: {
        ...state.userPrefs,
        userId: undefined,
      },
    }));
  },

  setAuthLoading: (loading) =>
    set((state) => ({
      auth: { ...state.auth, loading },
    })),

  setAuthError: (error) =>
    set((state) => ({
      auth: { ...state.auth, error },
    })),
}));

export default useStore;
