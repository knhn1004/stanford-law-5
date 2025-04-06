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
  // Ensure value is between 0 and 100
  const percentage = Math.min(Math.max(value, 0), 100);

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {label && <span className="text-xs w-24">{label}</span>}
      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${color}`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
      {showLabel && <span className="text-xs w-8">{percentage}%</span>}
    </div>
  );
}
