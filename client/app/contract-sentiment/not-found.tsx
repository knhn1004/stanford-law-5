import React from "react";
import Link from "next/link";

export default function NotFound() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex justify-between items-center mb-8">
        <div className="text-2xl font-bold text-blue-600">Clause Clarity</div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6 text-center">
        <div className="text-amber-500 text-xl mb-4">Analysis Not Found</div>
        <p className="mb-4">
          The contract analysis you&apos;re looking for doesn&apos;t exist or has expired.
        </p>
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
