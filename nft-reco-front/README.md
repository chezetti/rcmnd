# NFT Recommendation App

A modern web application for discovering and exploring NFTs with AI-powered recommendations.

## Features

- **Visual and Textual Search**: Find NFTs using images, descriptions or a combination of both
- **Advanced Filtering**: Filter NFTs by style, category, and other attributes
- **Recommendation Engine**: Discover similar NFTs based on your interests
- **Responsive Design**: Fully responsive interface that works on desktop and mobile devices
- **3D NFT Viewing**: View supported NFTs in 3D using Three.js
- **Masonry Layout**: Pinterest-style grid for aesthetic NFT browsing
- **User Preferences**: Personalized recommendations based on your interactions

## Tech Stack

- **Framework**: Next.js (App Router)
- **UI Library**: React
- **Styling**: TailwindCSS
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **Animations**: Framer Motion
- **3D Rendering**: React Three Fiber / drei
- **Layout**: react-masonry-css for masonry grid layouts
- **Form Handling**: React Hook Form
- **File Uploads**: Uppy

## Getting Started

### Prerequisites

- Node.js 16.8+ and npm/yarn

### Installation

1. Clone the repository

```bash
git clone <repository-url>
cd nft-reco-front
```

2. Install dependencies

```bash
npm install
# or
yarn
```

3. Start the development server

```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
nft-reco-front/
├── public/                # Static assets
├── src/
│   ├── app/               # Next.js App Router pages
│   │   ├── page.tsx       # Home/Explore page
│   │   ├── search/        # Search functionality
│   │   └── item/[id]/     # Individual NFT page
│   ├── components/        # React components
│   │   ├── layout/        # Layout components (header, footer)
│   │   ├── nft/           # NFT-specific components
│   │   └── ui/            # Reusable UI components
│   ├── lib/               # Utilities and services
│   │   ├── api.ts         # API service
│   │   └── store.ts       # Zustand store
│   └── globals.css        # Global styles
└── package.json           # Dependencies
```

## Key Components

### NFT Grid

The application uses two layout modes for displaying NFTs:

- **Grid Layout**: Traditional responsive grid
- **Masonry Layout**: Pinterest-style staggered grid for a more dynamic presentation

### NFT Card

Each NFT is displayed using a card component that shows:

- NFT image
- Name and collection
- Description
- Tags
- Like/favorite button
- View details button

### Search Interface

The search page allows users to:

- Upload a reference image
- Enter descriptive text
- Adjust balance between visual and textual search
- Set additional filters

## API Integration

The frontend integrates with a recommendation API that provides:

- NFT exploration with filters
- Similar item recommendations
- Search by image and text
- User feedback collection

## Deployment

The application can be built for production using:

```bash
npm run build
# or
yarn build
```

## License

[MIT License](LICENSE)
