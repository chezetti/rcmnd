export default function Footer() {
  return (
    <footer className="border-t border-border/40 bg-background/95">
      <div className="container flex flex-col items-center justify-between gap-4 py-6 md:py-8">
        <div className="flex flex-col items-center gap-4 px-4 md:px-0">
          <p className="text-center text-sm leading-loose text-muted-foreground">
            Built with Next.js, TailwindCSS, and shadcn/ui. Powered by
            multimodal embeddings.
          </p>
        </div>
        <p className="text-center text-sm text-muted-foreground">
          © {new Date().getFullYear()} NFT Recommender. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
