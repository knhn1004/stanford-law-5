"use client";

import React, { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { uploadContract, generateAnalysis } from "../services/contractAnalysis";

export default function FileUpload() {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileUpload = useCallback(async (file: File) => {
    if (file.type !== "application/pdf") {
      setError("Please upload a PDF document");
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const docId = await uploadContract(file);
      localStorage.setItem("contractDocId", docId);
      await generateAnalysis(docId);
      localStorage.setItem("contractAnalysisId", docId);
      router.push("/contract-sentiment");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during document processing");
      setIsUploading(false);
    }
  }, [router]);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  }, [handleFileUpload]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  }, [handleFileUpload]);

  const handleClick = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, []);

  return (
    <div
      className={`upload-area cursor-pointer ${
        isDragging ? "border-primary" : ""
      } ${isUploading ? "opacity-50" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <div className="flex flex-col items-center justify-center">
        <div className="mb-4 p-4 rounded-full bg-slate-50">
          <svg
            className="w-10 h-10 text-slate-700"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        </div>
        <div className="space-y-2 text-center">
          <p className="text-lg font-serif text-slate-800">
            Upload Legal Document
          </p>
          <p className="text-sm text-slate-600">
            Drag and drop your PDF document here, or click to browse
          </p>
          <p className="text-xs text-slate-500">
            Supported format: PDF
          </p>
        </div>
        {isUploading && (
          <div className="mt-6 w-full max-w-xs">
            <div className="relative pt-1">
              <div className="flex mb-2 items-center justify-between">
                <div>
                  <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-slate-700 bg-slate-100">
                    Processing
                  </span>
                </div>
              </div>
              <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-slate-100">
                <div
                  className="animate-pulse bg-slate-700"
                  style={{ width: "100%" }}
                ></div>
              </div>
            </div>
            <p className="text-sm text-slate-600 text-center">
              Analyzing document content...
            </p>
          </div>
        )}
        {error && (
          <div className="mt-4 text-sm text-rose-700 bg-rose-50 px-4 py-2 rounded">
            {error}
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".pdf"
          onChange={handleFileChange}
          disabled={isUploading}
        />
      </div>
    </div>
  );
}
