"use client";

import { ReactNode, useEffect } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import apiService from "@/lib/api";
import useStore from "@/lib/store";

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

export default function Providers({ children }: { children: ReactNode }) {
  const { loginSuccess, logout } = useStore();

  // Check for existing token and try to restore session
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("auth_token");
      if (token) {
        try {
          // Get current user data
          const response = await apiService.getCurrentUser();
          loginSuccess(response.data, token);
        } catch {
          // If the token is invalid or expired, clear it
          logout();
        }
      }
    };

    checkAuth();
  }, [loginSuccess, logout]);

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster />
    </QueryClientProvider>
  );
}
