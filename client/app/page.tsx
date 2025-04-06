import Image from "next/image";
import FileUpload from "./components/FileUpload";

export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-[32px] row-start-2 items-center w-full max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">ContractSentinel</h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload your contract for instant sentiment analysis and risk
            assessment
          </p>
        </div>

        <FileUpload />

        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500 mb-4">
            ContractSentinel analyzes your contracts for fairness, bias, and
            potential risks
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <svg
                className="w-5 h-5 text-green-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              Fairness Score
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <svg
                className="w-5 h-5 text-amber-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              Bias Detection
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <svg
                className="w-5 h-5 text-blue-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              Risk Assessment
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
