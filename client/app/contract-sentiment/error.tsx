"use client";

import React from "react";
import Link from "next/link";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex justify-between items-center mb-8">
        <div className="text-2xl font-bold text-blue-600">ContractSentinel</div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6 text-center">
        <div className="text-red-500 text-xl mb-4">Error</div>
        <p className="mb-4">
          Something went wrong while loading the contract analysis.
        </p>
        <p className="mb-4 text-red-500 text-sm font-mono border p-2 bg-gray-50 overflow-auto">
          {error.message}
        </p>
        <div className="flex justify-center gap-4">
          <button
            onClick={reset}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
          <Link
            href="/"
            className="bg-gray-200 text-gray-800 px-4 py-2 rounded hover:bg-gray-300 transition-colors"
          >
            Return to Home
          </Link>
        </div>
      </div>
    </div>
  );
}
