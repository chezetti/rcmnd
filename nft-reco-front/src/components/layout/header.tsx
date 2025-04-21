"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Menu, X, Search, User, LogOut } from "lucide-react";
import apiService from "@/lib/api";
import useStore, { NFTItem } from "@/lib/store";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function Header() {
  const pathname = usePathname();
  const router = useRouter();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState<NFTItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const searchTimeout = useRef<NodeJS.Timeout | null>(null);

  // Get auth state from store
  const { auth, logout } = useStore();
  const { isAuthenticated, user } = auth;

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    // Установить начальное состояние scrolled при монтировании
    handleScroll();

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    // Handle clicks outside of search results
    const handleClickOutside = (event: MouseEvent) => {
      if (
        searchRef.current &&
        !searchRef.current.contains(event.target as Node)
      ) {
        setShowResults(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleSearch = async (term: string) => {
    setSearchTerm(term);

    // Clear previous timeout
    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }

    if (term.trim().length < 2) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    // Debounce search requests
    searchTimeout.current = setTimeout(async () => {
      setIsSearching(true);
      try {
        const response = await apiService.explore({
          tags: term,
          limit: 6,
        });
        setSearchResults(response.data.results || []);
        setShowResults(true);
      } catch (error) {
        console.error("Search error:", error);
      } finally {
        setIsSearching(false);
      }
    }, 300);
  };

  const handleSearchItemClick = (uuid: string) => {
    router.push(`/item/${uuid}`);
    setShowResults(false);
    setSearchTerm("");
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchTerm.trim().length > 0) {
      router.push(`/search?q=${encodeURIComponent(searchTerm)}`);
      setShowResults(false);
    }
  };

  const handleLogout = () => {
    logout();
    router.push("/");
  };

  const navItems = [
    { name: "Explore", path: "/" },
    { name: "Search", path: "/search" },
    { name: "Statistics", path: "/statistics" },
    // Only show Upload and Dashboard tabs for authenticated users
    ...(isAuthenticated
      ? [
          { name: "Upload", path: "/upload" },
          { name: "Dashboard", path: "/dashboard" },
        ]
      : []),
  ];

  return (
    <header
      className={`sticky top-0 z-50 w-full transition-all duration-300 ${
        scrolled
          ? "scrolled bg-background/80 backdrop-blur-xl shadow-sm"
          : "bg-background"
      }`}
    >
      <div className="container-fluid py-3 flex items-center justify-between px-4 md:px-8 mx-auto">
        <div className="flex items-center">
          <Link href="/" className="flex items-center space-x-3">
            <span className="font-bold text-2xl tracking-wide">
              <span className="bg-gradient-to-r from-purple-600 via-pink-400 to-orange-600 bg-clip-text text-transparent">
                RCMND
              </span>
            </span>
          </Link>

          {/* Search box */}
          <div
            ref={searchRef}
            className="hidden md:flex ml-8 relative rounded-full"
          >
            <form onSubmit={handleSearchSubmit} className="relative">
              <div className="flex items-center w-72 h-10 px-4 border border-input rounded-full bg-background">
                <Search className="h-4 w-4 text-muted-foreground mr-2" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => handleSearch(e.target.value)}
                  placeholder="Search NFTs..."
                  className="bg-transparent border-none outline-none w-full text-sm"
                />
                {isSearching && (
                  <div className="w-4 h-4 border-t-2 border-primary rounded-full animate-spin mr-2"></div>
                )}
              </div>

              {/* Search results dropdown */}
              {showResults && searchResults.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-card border border-border rounded-lg shadow-lg overflow-hidden z-50">
                  <div className="max-h-[400px] overflow-y-auto py-2">
                    {searchResults.map((item) => (
                      <div
                        key={item.uuid}
                        className="px-4 py-2 hover:bg-muted cursor-pointer"
                        onClick={() => handleSearchItemClick(item.uuid)}
                      >
                        <div className="flex items-center">
                          <div className="w-8 h-8 bg-muted rounded overflow-hidden mr-3">
                            <Image
                              src={`https://picsum.photos/seed/${item.uuid}/32/32`}
                              alt={item.name || "NFT"}
                              width={32}
                              height={32}
                              className="object-cover"
                              unoptimized
                            />
                          </div>
                          <div>
                            <p className="font-medium text-sm">
                              {item.name || "Unnamed NFT"}
                            </p>
                            <p className="text-xs text-muted-foreground truncate max-w-[200px]">
                              {item.description?.substring(0, 30) ||
                                "No description"}
                              {(item.description?.length || 0) > 30
                                ? "..."
                                : ""}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="border-t border-gray-600 border-border px-4 py-2">
                    <Button
                      variant="link"
                      className="text-xs w-full justify-center text-primary"
                      onClick={() =>
                        router.push(
                          `/search?q=${encodeURIComponent(searchTerm)}`
                        )
                      }
                    >
                      See all results
                    </Button>
                  </div>
                </div>
              )}
            </form>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center ml-6 space-x-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  pathname === item.path
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-secondary"
                }`}
              >
                {item.name}
              </Link>
            ))}
          </nav>
        </div>

        {/* Right side actions */}
        <div className="flex items-center space-x-3">
          {isAuthenticated ? (
            <div className="hidden md:flex items-center space-x-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-2">
                    <User className="h-4 w-4" />
                    <span>{user?.username}</span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => router.push("/dashboard")}>
                    Dashboard
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => router.push("/profile")}>
                    Profile
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleLogout}>
                    <LogOut className="h-4 w-4 mr-2" />
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          ) : (
            <div className="hidden md:block">
              <Link href="/login">
                <div className="p-[1px] bg-gradient-to-r from-purple-600 via-pink-400 to-orange-600 overflow-hidden rounded-lg">
                  <Button
                    size="lg"
                    className="bg-background text-foreground hover:bg-background/90 rounded-md"
                  >
                    LOGIN
                  </Button>
                </div>
              </Link>
            </div>
          )}

          {/* Mobile menu button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={toggleMenu}
            aria-label="Toggle menu"
          >
            {isMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </Button>
        </div>
      </div>

      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden border-t border-border/10">
          <div className="container-fluid px-4 py-4">
            <nav className="flex flex-col space-y-4">
              {/* Mobile search */}
              <form onSubmit={handleSearchSubmit}>
                <div className="flex items-center w-full h-10 px-4 border border-input rounded-full bg-background">
                  <Search className="h-4 w-4 text-muted-foreground mr-2" />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => handleSearch(e.target.value)}
                    placeholder="Search NFTs..."
                    className="bg-transparent border-none outline-none w-full text-sm"
                  />
                  {isSearching && (
                    <div className="w-4 h-4 border-t-2 border-primary rounded-full animate-spin mr-2"></div>
                  )}
                </div>
              </form>

              {navItems.map((item) => (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`px-4 py-2 rounded-lg text-sm font-medium ${
                    pathname === item.path
                      ? "bg-primary text-primary-foreground"
                      : "text-foreground"
                  }`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.name}
                </Link>
              ))}

              {/* Mobile auth options */}
              <div className="pt-2">
                {isAuthenticated ? (
                  <>
                    <div className="border-t border-border/10 pt-3 mb-2">
                      <p className="px-4 text-sm text-muted-foreground">
                        Signed in as{" "}
                        <span className="font-medium">{user?.username}</span>
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      className="w-full justify-start"
                      onClick={() => {
                        handleLogout();
                        setIsMenuOpen(false);
                      }}
                    >
                      <LogOut className="h-4 w-4 mr-2" />
                      Logout
                    </Button>
                  </>
                ) : (
                  <div className="rounded-lg p-[1px] bg-gradient-to-r from-purple-600 via-pink-400 to-orange-600 overflow-hidden group">
                    <Link href="/login" onClick={() => setIsMenuOpen(false)}>
                      <button className="w-full px-4 py-2 bg-background text-foreground font-medium rounded-md text-sm transition-transform transform active:scale-95">
                        LOGIN
                        <span className="absolute inset-0 bg-gradient-to-r from-pink-500 to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                      </button>
                    </Link>
                  </div>
                )}
              </div>
            </nav>
          </div>
        </div>
      )}
    </header>
  );
}
