import React from "react";

interface ProgressBarProps {
  value: number;
  color?: string;
  showLabel?: boolean;
  label?: string;
  className?: string;
}

export default function ProgressBar({
  value,
  color = "bg-blue-500",
  showLabel = true,
  label,
  className = "",
}: ProgressBarProps) {
  // Force the percentage to be at least 20% if a value exists, for visual clarity
  // This is a workaround to ensure progress is always visible
  const percentage = value ? Math.max(Math.min(Math.max(value, 20), 100), 20) : 0;

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {label && <span className="text-xs w-24">{label}</span>}
      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${color}`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
      {showLabel && <span className="text-xs w-8">{value}%</span>}
    </div>
  );
}
