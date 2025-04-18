"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";

export default function Header() {
  const pathname = usePathname();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const navItems = [
    { name: "Explore", path: "/" },
    { name: "Search", path: "/search" },
    { name: "Upload", path: "/upload" },
    { name: "Dashboard", path: "/dashboard" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-md shadow-sm supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center">
          <Link href="/" className="flex items-center space-x-2">
            <span className="font-bold text-xl bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-pink-500">
              NFT Recommender
            </span>
          </Link>
        </div>

        {/* Мобильное меню кнопка */}
        <button
          className="md:hidden p-2"
          onClick={toggleMenu}
          aria-label="Toggle menu"
        >
          {isMenuOpen ? (
            <X className="h-6 w-6" />
          ) : (
            <Menu className="h-6 w-6" />
          )}
        </button>

        {/* Десктопное меню */}
        <div className="hidden md:flex items-center space-x-6">
          <nav className="flex items-center space-x-6 text-sm font-medium">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path}
                className={`transition-colors hover:text-foreground/80 ${
                  pathname === item.path
                    ? "text-foreground font-semibold"
                    : "text-foreground/60"
                }`}
              >
                {item.name}
              </Link>
            ))}
          </nav>
          <Button variant="outline" size="sm" asChild>
            <Link href="/auth/login">Login</Link>
          </Button>
        </div>

        {/* Мобильное меню выпадающее */}
        {isMenuOpen && (
          <div className="absolute top-16 left-0 right-0 bg-background/95 backdrop-blur-md border-b border-border z-50 md:hidden shadow-md">
            <div className="container py-4">
              <nav className="flex flex-col space-y-4">
                {navItems.map((item) => (
                  <Link
                    key={item.path}
                    href={item.path}
                    className={`transition-colors hover:text-foreground/80 ${
                      pathname === item.path
                        ? "text-foreground font-semibold"
                        : "text-foreground/60"
                    }`}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    {item.name}
                  </Link>
                ))}
                <Button variant="outline" size="sm" className="w-full" asChild>
                  <Link href="/auth/login">Login</Link>
                </Button>
              </nav>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
