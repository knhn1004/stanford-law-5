import React from "react";

export type SentimentType =
  | "positive"
  | "negative"
  | "neutral"
  | "biased-buyer"
  | "biased-seller"
  | "high"
  | "medium"
  | "low";

interface SentimentBadgeProps {
  type: SentimentType;
  label?: string;
}

export default function SentimentBadge({ type, label }: SentimentBadgeProps) {
  const baseClasses = "sentiment-badge";
  
  const typeClasses = {
    "positive": "sentiment-positive",
    "negative": "sentiment-negative",
    "neutral": "sentiment-neutral",
    "biased-buyer": "sentiment-biased-buyer",
    "biased-seller": "sentiment-biased-seller",
    "high": "sentiment-negative",
    "medium": "sentiment-neutral",
    "low": "sentiment-positive"
  };

  return (
    <span className={`${baseClasses} ${typeClasses[type]}`}>
      {label || type}
    </span>
  );
}
