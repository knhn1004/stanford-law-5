import React from "react";

type SentimentType =
  | "positive"
  | "negative"
  | "neutral"
  | "biased-buyer"
  | "biased-seller";

interface SentimentBadgeProps {
  type: SentimentType;
  label: string;
}

export default function SentimentBadge({ type, label }: SentimentBadgeProps) {
  const getBadgeClass = () => {
    switch (type) {
      case "positive":
        return "sentiment-positive";
      case "negative":
        return "sentiment-negative";
      case "neutral":
        return "sentiment-neutral";
      case "biased-buyer":
        return "sentiment-biased-buyer";
      case "biased-seller":
        return "sentiment-biased-seller";
      default:
        return "sentiment-neutral";
    }
  };

  return <span className={`sentiment-badge ${getBadgeClass()}`}>{label}</span>;
}
