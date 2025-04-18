import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function SkeletonCard() {
  return (
    <Card className="h-full flex flex-col overflow-hidden">
      <div className="relative aspect-square overflow-hidden bg-muted">
        <Skeleton className="h-full w-full" />
      </div>
      <CardContent className="flex-grow p-4">
        <div className="flex justify-between items-start">
          <Skeleton className="h-5 w-3/4 mb-2" />
          <Skeleton className="h-4 w-10 rounded-full" />
        </div>
        <Skeleton className="h-4 w-1/3 mt-2 mb-1" />
        <Skeleton className="h-3 w-full mt-2" />
        <Skeleton className="h-3 w-4/5 mt-1 mb-3" />
        <div className="flex gap-1 mt-3">
          <Skeleton className="h-5 w-12 rounded-full" />
          <Skeleton className="h-5 w-14 rounded-full" />
          <Skeleton className="h-5 w-10 rounded-full" />
        </div>
      </CardContent>
      <CardFooter className="p-4 pt-0 flex justify-between">
        <Skeleton className="h-8 w-24 rounded-full" />
        <Skeleton className="h-8 w-8 rounded-full" />
      </CardFooter>
    </Card>
  );
}
