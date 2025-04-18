import * as React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface FilterSelectProps {
  placeholder: string;
  value: string;
  onValueChange: (value: string) => void;
  options: { value: string; label: string }[];
  className?: string;
}

/**
 * A consistent FilterSelect component to be used across the application
 * for all filter dropdowns. Ensures consistent styling and behavior.
 */
export default function FilterSelect({
  placeholder,
  value,
  onValueChange,
  options,
  className,
}: FilterSelectProps) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger className={className}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="bg-card border-border shadow-md">
        {options.map((option) => (
          <SelectItem
            key={option.value}
            value={option.value}
            className="focus:bg-primary/10"
          >
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
