import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";

import { cn } from "@/lib/utils";

interface SliderProps
  extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> {
  variant?: "default" | "balance";
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ className, variant = "default", ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none select-none items-center",
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track
      className={cn(
        "relative h-2 w-full grow overflow-hidden rounded-full bg-secondary",
        variant === "balance" &&
          "bg-gradient-to-r from-blue-500 via-slate-700 to-purple-500"
      )}
    >
      <SliderPrimitive.Range
        className={cn(
          "absolute h-full bg-primary",
          variant === "balance" && "opacity-0"
        )}
      />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb
      className={cn(
        "block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background shadow-md hover:shadow-lg transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        variant === "balance" &&
          "border-indigo-500 hover:scale-110 bg-white z-10"
      )}
    />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
