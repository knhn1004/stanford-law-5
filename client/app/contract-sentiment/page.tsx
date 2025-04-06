"use client";

import React, { useEffect, useState } from "react";
import SentimentBadge from "../components/SentimentBadge";
import ProgressBar from "../components/ProgressBar";
import { getContractAnalysis } from "../services/contractAnalysis";
import Link from "next/link";

// Define types for the contract analysis data
interface BiasIndicator {
  label: string;
  value: number;
}

interface NotableClause {
  type: string;
  sentiment: string;
  sentimentLabel: string;
  biasScore: number;
  riskLevel: string;
  riskLabel: string;
  text: string;
  analysis: string;
  biasIndicators: BiasIndicator[];
  industryComparison: string;
  recommendations: string[];
}

interface SummaryPoint {
  title: string;
  description: string;
}

interface ContractAnalysis {
  id: string;
  timestamp: string;
  contractName: string;
  description: string;
  metrics: {
    overallFairnessScore: number;
    potentialBiasIndicators: number;
    highRiskClauses: number;
    balancedClauses: number;
  };
  sentimentDistribution: {
    vendorFavorable: number;
    balanced: number;
    customerFavorable: number;
    neutral: number;
  };
  notableClauses: NotableClause[];
  industryBenchmarking: {
    fairnessScore: number;
    percentile: number;
    summary: string;
  };
  summary: {
    title: string;
    description: string;
    points: SummaryPoint[];
    riskAssessment: {
      level: string;
      label: string;
      description: string;
    };
  };
}

export default function ContractSentimentPage() {
  const [analysis, setAnalysis] = useState<ContractAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        // Get the analysis ID from localStorage
        const analysisId = localStorage.getItem("contractAnalysisId");

        if (!analysisId) {
          setError("No analysis ID found. Please upload a contract first.");
          setLoading(false);
          return;
        }

        // Fetch the analysis data
        const data = await getContractAnalysis(analysisId);
        setAnalysis(data as ContractAnalysis);
      } catch (err) {
        setError("Failed to load contract analysis. Please try again.");
        console.error("Error fetching analysis:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, []);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="flex justify-between items-center mb-8">
          <div className="text-2xl font-bold text-blue-600">
            ContractSentinel
          </div>
          <div className="text-sm text-gray-500">Loading analysis...</div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded w-3/4 mb-4"></div>
            <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="bg-white rounded-lg shadow-md p-5 text-center"
            >
              <div className="animate-pulse">
                <div className="h-8 bg-gray-200 rounded w-16 mx-auto mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-24 mx-auto"></div>
              </div>
            </div>
          ))}
        </div>

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="animate-pulse">
            <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
            <div className="h-64 bg-gray-200 rounded mb-4"></div>
            <div className="flex justify-center gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-4 bg-gray-200 rounded w-24"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="flex justify-between items-center mb-8">
          <div className="text-2xl font-bold text-blue-600">
            ContractSentinel
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6 mb-6 text-center">
          <div className="text-red-500 text-xl mb-4">Error</div>
          <p className="mb-4">{error}</p>
          <Link
            href="/"
            className="inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
          >
            Return to Home
          </Link>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return null;
  }

  // Format the timestamp
  const formattedDate = new Date(analysis.timestamp).toLocaleString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex justify-between items-center mb-8">
        <div className="text-2xl font-bold text-blue-600">ContractSentinel</div>
        <div className="text-sm text-gray-500">Analyzed: {formattedDate}</div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h1 className="text-2xl font-bold mb-2">{analysis.contractName}</h1>
        <p className="text-gray-600">{analysis.description}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow-md p-5 text-center">
          <div className="text-3xl font-bold text-blue-600 mb-1">
            {analysis.metrics.overallFairnessScore}
          </div>
          <div className="text-sm text-gray-500">Overall Fairness Score</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-5 text-center">
          <div className="text-3xl font-bold text-amber-500 mb-1">
            {analysis.metrics.potentialBiasIndicators}
          </div>
          <div className="text-sm text-gray-500">Potential Bias Indicators</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-5 text-center">
          <div className="text-3xl font-bold text-red-500 mb-1">
            {analysis.metrics.highRiskClauses}
          </div>
          <div className="text-sm text-gray-500">High-Risk Clauses</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-5 text-center">
          <div className="text-3xl font-bold text-green-500 mb-1">
            {analysis.metrics.balancedClauses}
          </div>
          <div className="text-sm text-gray-500">Balanced Clauses</div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200">
          Sentiment Distribution by Clause Type
        </h2>
        <div className="h-80 mb-4">
          {/* Placeholder for chart - would use a charting library in production */}
          <div className="w-full h-full bg-gray-100 rounded flex items-center justify-center">
            <p className="text-gray-500">Sentiment Distribution Chart</p>
          </div>
        </div>
        <div className="flex flex-wrap justify-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-sm">
              Vendor-favorable ({analysis.sentimentDistribution.vendorFavorable}
              %)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm">
              Balanced ({analysis.sentimentDistribution.balanced}%)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-sm">
              Customer-favorable (
              {analysis.sentimentDistribution.customerFavorable}%)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
            <span className="text-sm">
              Neutral ({analysis.sentimentDistribution.neutral}%)
            </span>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200">
          Notable Clauses with Sentiment Analysis
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="py-3 px-4 text-left text-sm font-medium text-gray-500">
                  Clause Type
                </th>
                <th className="py-3 px-4 text-left text-sm font-medium text-gray-500">
                  Sentiment
                </th>
                <th className="py-3 px-4 text-left text-sm font-medium text-gray-500">
                  Bias Score
                </th>
                <th className="py-3 px-4 text-left text-sm font-medium text-gray-500">
                  Risk Level
                </th>
              </tr>
            </thead>
            <tbody>
              {analysis.notableClauses.map((clause, index) => (
                <tr
                  key={index}
                  className={
                    index < analysis.notableClauses.length - 1
                      ? "border-b border-gray-100"
                      : ""
                  }
                >
                  <td className="py-3 px-4">{clause.type}</td>
                  <td className="py-3 px-4">
                    <SentimentBadge
                      type={clause.sentiment as any}
                      label={clause.sentimentLabel}
                    />
                  </td>
                  <td className="py-3 px-4">
                    <ProgressBar value={clause.biasScore} color="bg-red-500" />
                  </td>
                  <td className="py-3 px-4">
                    <SentimentBadge
                      type={clause.riskLevel as any}
                      label={clause.riskLabel}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200">
          Detailed Clause Analysis
        </h2>

        {analysis.notableClauses.slice(0, 2).map((clause, index) => (
          <div key={index} className={index > 0 ? "mt-8" : ""}>
            <h3 className="text-lg font-medium mb-2">
              {clause.type} (Section {index + 1}.{index + 2})
            </h3>
            <div className="bg-gray-50 p-4 rounded border-l-4 border-blue-400 text-sm mb-4 whitespace-pre-wrap">
              {clause.text}
            </div>

            <div className="mb-4">
              <div className="font-medium mb-2">Sentiment Analysis:</div>
              <p className="mb-2">{clause.analysis}</p>
              <ul className="list-disc pl-5 mb-4">
                {clause.recommendations.slice(0, 4).map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>

            <div className="flex flex-col md:flex-row gap-8 mb-4">
              <div className="flex-1">
                <div className="font-medium mb-2">Bias Indicators:</div>
                {clause.biasIndicators.map((indicator, i) => (
                  <ProgressBar
                    key={i}
                    value={indicator.value}
                    color={
                      i === 0
                        ? "bg-red-500"
                        : i === 1
                        ? "bg-blue-500"
                        : "bg-gray-500"
                    }
                    label={indicator.label}
                    className="mb-2"
                  />
                ))}
              </div>

              <div className="flex-1">
                <div className="font-medium mb-2">Industry Comparison:</div>
                <p className="text-sm">{clause.industryComparison}</p>
              </div>
            </div>

            <div className="bg-blue-50 p-4 rounded border-l-4 border-blue-500">
              <div className="font-medium mb-2">Recommended Changes:</div>
              <ol className="list-decimal pl-5">
                {clause.recommendations.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ol>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200">
          Industry Benchmarking
        </h2>
        <div className="h-96 mb-4">
          {/* Placeholder for chart - would use a charting library in production */}
          <div className="w-full h-full bg-gray-100 rounded flex items-center justify-center">
            <p className="text-gray-500">Industry Comparison Chart</p>
          </div>
        </div>
        <p>{analysis.industryBenchmarking.summary}</p>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200">
          {analysis.summary.title}
        </h2>
        <p className="mb-4">{analysis.summary.description}</p>

        <ol className="list-decimal pl-5 mb-4">
          {analysis.summary.points.map((point, index) => (
            <li key={index}>
              <strong>{point.title}:</strong> {point.description}
            </li>
          ))}
        </ol>

        <p className="mt-4">
          <strong>Overall Risk Assessment:</strong>{" "}
          {analysis.summary.riskAssessment.description}
        </p>
      </div>
    </div>
  );
}
