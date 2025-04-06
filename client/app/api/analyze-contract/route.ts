import { NextRequest, NextResponse } from 'next/server';

// Mock function to simulate contract analysis
async function analyzeContract(file: File): Promise<string> {
  // In a real implementation, this would send the file to the FastAPI backend
  // For now, we'll just return a mock analysis ID
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve('mock-analysis-123');
    }, 1500); // Simulate processing time
  });
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }
    
    // Check if file is a PDF
    if (file.type !== 'application/pdf') {
      return NextResponse.json(
        { error: 'Only PDF files are supported' },
        { status: 400 }
      );
    }
    
    // Analyze the contract
    const analysisId = await analyzeContract(file);
    
    return NextResponse.json({ analysisId });
  } catch (error) {
    console.error('Error analyzing contract:', error);
    return NextResponse.json(
      { error: 'Failed to analyze contract' },
      { status: 500 }
    );
  }
} 