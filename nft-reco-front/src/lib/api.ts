import axios from "axios";

// Используем API прокси через Next.js для обхода CORS
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
export const apiService = {
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

  // Get health status
  getHealth: () => api.get("/health"),

  // Get stats
  getStats: () => api.get("/stats"),

  // Submit feedback
  submitFeedback: (data: {
    item_uuid: string;
    user_id?: string;
    feedback_type: "click" | "view" | "favorite" | "purchase" | "relation";
    value?: number;
    related_uuid?: string;
  }) => api.post("/feedback", data),

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
