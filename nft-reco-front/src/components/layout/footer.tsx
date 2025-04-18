"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Footer() {
  const [showCookieNotice, setShowCookieNotice] = useState(false);

  useEffect(() => {
    // Check if the cookie notice has been accepted before
    const cookieAccepted = localStorage.getItem("cookieNoticeAccepted");
    if (!cookieAccepted) {
      setShowCookieNotice(true);
    }
  }, []);

  const handleCloseCookieNotice = () => {
    setShowCookieNotice(false);
    // Save in localStorage to remember the user's choice
    localStorage.setItem("cookieNoticeAccepted", "true");
  };

  return (
    <footer className="bg-transparent text-[#f3f4f6] pt-6">
      {/* 16px по бокам на XS, 32px на MD+ */}
      <div className="mx-4 md:mx-8 border-t border-gray-600" />

      {/* Cookie notice with animation */}
      <AnimatePresence>
        {showCookieNotice && (
          <motion.div
            className="fixed bottom-6 left-6 z-50 max-w-sm bg-card rounded-lg p-4 shadow-lg border border-border/60 flex items-center justify-between"
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <p className="text-sm font-medium mr-4">
              We eat use cookies for better experience
            </p>
            <motion.button
              onClick={handleCloseCookieNotice}
              className="text-muted-foreground"
              whileHover={{
                scale: 1.1,
                rotate: 90,
                color: "white",
              }}
              transition={{ duration: 0.2 }}
            >
              <X className="h-4 w-4" />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="w-full max-w-[1300px] mx-auto px-4 py-4">
        <div className="flex items-center justify-center text-center gap-8">
          <div className="text-center">
            <p className="text-sm text-gray-500 font-medium">
              © {new Date().getFullYear()} RCMND, Inc.
            </p>
          </div>

          <Link
            href="/community"
            className="text-sm text-gray-500 hover:text-[#f3f4f6] transition-colors font-medium"
          >
            Community guidelines
          </Link>

          <Link
            href="/terms"
            className="text-sm text-gray-500 hover:text-[#f3f4f6] transition-colors font-medium"
          >
            Terms
          </Link>

          <Link
            href="/privacy"
            className="text-sm text-gray-500 hover:text-[#f3f4f6] transition-colors font-medium"
          >
            Privacy policy
          </Link>
        </div>
      </div>
    </footer>
  );
}
