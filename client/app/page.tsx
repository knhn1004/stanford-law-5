import FileUpload from "./components/FileUpload";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16 sm:py-24">
        <div className="text-center space-y-8 mb-16">
          <div className="space-y-4">
            <h1 className="text-5xl sm:text-6xl font-serif font-bold text-slate-800">
              Clause Clarity
            </h1>
            <p className="text-xl sm:text-2xl text-slate-600 max-w-2xl mx-auto leading-relaxed font-serif">
              Contract Analysis for Individuals and SMBs
            </p>
            <p className="text-lg text-slate-500 max-w-2xl mx-auto">
              Upload your contract for comprehensive analysis of fairness, risk, and compliance
            </p>
          </div>
          
          <div className="flex flex-wrap justify-center gap-4 text-sm">
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-lg shadow-sm border border-slate-200">
              <svg
                className="w-5 h-5 text-emerald-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              <span className="font-medium text-slate-800">Fairness Analysis</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-lg shadow-sm border border-slate-200">
              <svg
                className="w-5 h-5 text-amber-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <span className="font-medium text-slate-800">Risk Detection</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-lg shadow-sm border border-slate-200">
              <svg
                className="w-5 h-5 text-blue-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className="font-medium text-slate-800">Compliance Check</span>
            </div>
          </div>
        </div>

        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg border border-slate-100 p-8">
            <FileUpload />
          </div>
        </div>

        <div className="mt-16 text-center">
          <p className="text-slate-600 max-w-2xl mx-auto font-serif leading-relaxed">
            Our advanced AI-powered platform analyzes legal documents with precision, 
            helping legal professionals identify potential risks, ensure compliance, 
            and maintain the highest standards of legal practice.
          </p>
        </div>
      </div>
    </div>
  );
}

