import axios from "axios";

// Используем API прокси через Next.js для обхода CORS
// Важно: в бэкенде маршруты имеют префикс /api, поэтому его не нужно добавлять здесь,
// так как Next.js rewrites уже перенаправляют запросы правильно
const API_URL = "/api"; // Это будет проксироваться через Next.js rewrites

// Create axios instance with base URL and default headers
const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
  // Настройки для работы с CORS
  withCredentials: true,
});

// Add a request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    // Get token from local storage or session
    const token =
      typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// API service functions
const apiService = {
  // Authentication operations
  register: (userData: {
    username: string;
    email: string;
    full_name?: string;
    password: string;
  }) => api.post("/auth/register", userData),

  login: (loginData: { username: string; password: string }) =>
    api.post("/auth/login", loginData),

  getCurrentUser: () => api.get("/auth/me"),

  // Item operations
  getItem: (uuid: string) => api.get(`/items/${uuid}`),
  deleteItem: (uuid: string) => api.delete(`/items/${uuid}`),

  // Explore NFTs with filtering
  explore: (params: {
    category?: string;
    style?: string;
    tags?: string;
    limit?: number;
    offset?: number;
    sort_by?: string;
  }) => api.get("/explore", { params }),

  // Get user's favorite items
  getUserFavorites: (params?: { limit?: number; offset?: number }) =>
    api.get("/user/favorites", { params }),

  // Get favorite items by user ID
  getUserFavoritesById: (
    userId: string,
    params?: { limit?: number; offset?: number }
  ) => api.get(`/users/${userId}/favorites`, { params }),

  // Get health status
  getHealth: () => api.get("/health"),

  // Get stats
  getStats: () => api.get("/stats"),

  // Get all stats (global)
  getAllStats: () => api.get("/stats/all"),

  // Get trending NFTs
  getTrendingNFTs: (params?: { limit?: number; offset?: number }) =>
    api.get("/trending", { params }),

  // Submit feedback
  submitFeedback: (data: {
    item_uuid: string;
    user_id?: string;
    feedback_type: "click" | "view" | "favorite" | "purchase" | "relation";
    value?: number;
    related_uuid?: string;
  }) => {
    console.log(
      `Submitting ${data.feedback_type} feedback for item ${data.item_uuid}`
    );
    return api.post("/feedback", { feedback: data }).catch((error) => {
      console.error(`Failed to submit ${data.feedback_type} feedback:`, error);
      throw error;
    });
  },

  // Get recommendations
  getRecommendations: (formData: FormData) =>
    api.post("/recommend", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }),

  // Upload a new NFT
  uploadNFT: (formData: FormData) =>
    api.post("/items", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }),
};

export default apiService;
