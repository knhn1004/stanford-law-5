// Mock data for contract analysis
// Removed unused mock data

// Define types for contract analysis data
export interface BiasIndicator {
	label: string;
	value: number;
}

export interface NotableClause {
	type: string;
	sentiment:
		| 'positive'
		| 'negative'
		| 'neutral'
		| 'biased-buyer'
		| 'biased-seller';
	sentimentLabel: string;
	biasScore: number;
	riskLevel:
		| 'positive'
		| 'negative'
		| 'neutral'
		| 'biased-buyer'
		| 'biased-seller';
	riskLabel: string;
	text: string;
	analysis: string;
	biasIndicators: BiasIndicator[];
	industryComparison: string;
	recommendations: string[];
}

export interface SummaryPoint {
	title: string;
	description: string;
}

export interface ContractAnalysis {
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
	notableClauses: Array<{
		type: string;
		sentiment: string;
		sentimentLabel: string;
		biasScore: number;
		riskLevel: string;
		riskLabel: string;
		text: string;
		analysis: string;
		biasIndicators: Array<{
			label: string;
			value: number;
		}>;
		industryComparison: string;
		recommendations: string[];
		legalPrecedents: Array<{
			case: string;
			relevance: string;
			implication: string;
		}>;
	}>;
	industryBenchmarking: {
		fairnessScore: number;
		percentile: number;
		summary: string;
	};
	summary: {
		title: string;
		description: string;
		points: Array<{
			title: string;
			description: string;
		}>;
		riskAssessment: {
			level: string;
			label: string;
			description: string;
		};
	};
	legalReferences: Array<{
		title: string;
		description: string;
		url: string;
	}>;
	industryStandards: Array<{
		name: string;
		description: string;
		complianceStatus: string;
	}>;
	regulatoryGuidelines: Array<{
		regulation: string;
		relevance: string;
		complianceStatus: string;
	}>;
}

// Function to upload a contract to FastAPI server
export async function uploadContract(file: File) {
	try {
		const formData = new FormData();
		formData.append('file', file);

		const response = await fetch('http://localhost:8000/upload', {
			method: 'POST',
			body: formData,
		});

		if (!response.ok) {
			throw new Error(`Error uploading contract: ${response.statusText}`);
		}

		const data = await response.json();
		return data.doc_id;
	} catch (error) {
		console.error('Error uploading contract:', error);
		throw error;
	}
}

// Function to generate analysis using RAG with Ollama
export async function generateAnalysis(docId: string) {
	try {
		const response = await fetch('http://localhost:8000/query', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				query: `Analyze this contract for sentiment, bias, and fairness. 
    Focus on identifying vendor-favorable vs customer-favorable clauses.
    Provide an overall fairness score out of 100, risk assessment, and recommendations.`,
				doc_id: docId,
			}),
		});

		if (!response.ok) {
			throw new Error(`Error generating analysis: ${response.statusText}`);
		}

		const responseData = await response.json();

		try {
			// The response should already be valid JSON from the server
			const parsedResponse = JSON.parse(responseData.response);
			
			// Create the analysis data from the parsed response
			const analysisData = {
				id: docId,
				timestamp: new Date().toISOString(),
				contractName: parsedResponse.contractName || `Contract ${docId}`,
				description: typeof parsedResponse.description === 'string' 
					? parsedResponse.description 
					: 'Analysis of the contract',
				metrics: parsedResponse.metrics || {
					overallFairnessScore: 50,
					potentialBiasIndicators: 0,
					highRiskClauses: 0,
					balancedClauses: 0,
				},
				sentimentDistribution: parsedResponse.sentimentDistribution || {
					vendorFavorable: 25,
					balanced: 25,
					customerFavorable: 25,
					neutral: 25,
				},
				notableClauses: Array.isArray(parsedResponse.notableClauses) 
					? parsedResponse.notableClauses.map(clause => ({
						...clause,
						text: typeof clause.text === 'string' ? clause.text : 'Contract clause text unavailable',
						analysis: typeof clause.analysis === 'string' ? clause.analysis : clause.analysis,
						industryComparison: typeof clause.industryComparison === 'string' ? clause.industryComparison : clause.industryComparison,
						recommendations: Array.isArray(clause.recommendations)
							? clause.recommendations
							: [],
					}))
					: [],
				industryBenchmarking: {
					fairnessScore: parsedResponse.metrics?.overallFairnessScore || 50,
					percentile: 50,
					summary: typeof parsedResponse.summary?.description === 'string' 
						? parsedResponse.summary.description.substring(0, 200) 
						: 'Analysis based on contract content.',
				},
				summary: {
					title: parsedResponse.summary?.title || 'Analysis Summary',
					description: typeof parsedResponse.summary?.description === 'string'
						? parsedResponse.summary.description
						: 'Contract analysis summary',
					points: Array.isArray(parsedResponse.summary?.points)
						? parsedResponse.summary.points.map(point => ({
							title: point.title || '',
							description: typeof point.description === 'string' ? point.description : '',
						}))
						: [],
					riskAssessment: parsedResponse.summary?.riskAssessment || {
						level: 'neutral',
						label: 'Medium Risk',
						description: 'Risk assessment based on contract analysis.',
					},
				},
				// Add missing properties for legal references and standards
				legalReferences: Array.isArray(parsedResponse.legalReferences)
					? parsedResponse.legalReferences.map(ref => ({
						title: ref.title || '',
						description: typeof ref.description === 'string' ? ref.description : '',
						url: typeof ref.url === 'string' ? ref.url : '#',
					}))
					: [],
				industryStandards: Array.isArray(parsedResponse.industryStandards)
					? parsedResponse.industryStandards.map(standard => ({
						name: standard.name || '',
						description: typeof standard.description === 'string' ? standard.description : '',
						complianceStatus: standard.complianceStatus || 'Unknown',
					}))
					: [],
				regulatoryGuidelines: Array.isArray(parsedResponse.regulatoryGuidelines)
					? parsedResponse.regulatoryGuidelines.map(guideline => ({
						regulation: guideline.regulation || '',
						relevance: typeof guideline.relevance === 'string' ? guideline.relevance : '',
						complianceStatus: guideline.complianceStatus || 'Unknown',
					}))
					: [],
			};

			// Store the analysis data in localStorage for later retrieval
			localStorage.setItem(
				`contractAnalysis_${docId}`,
				JSON.stringify(analysisData)
			);

			return analysisData;
		} catch (error) {
			console.error('Error parsing analysis response:', error);
			// Return a default analysis structure
			const defaultAnalysis = {
				id: docId,
				timestamp: new Date().toISOString(),
				contractName: `Contract ${docId}`,
				description: 'Error processing contract analysis.',
				metrics: {
					overallFairnessScore: 50,
					potentialBiasIndicators: 0,
					highRiskClauses: 0,
					balancedClauses: 0,
				},
				sentimentDistribution: {
					vendorFavorable: 25,
					balanced: 25,
					customerFavorable: 25,
					neutral: 25,
				},
				notableClauses: [],
				industryBenchmarking: {
					fairnessScore: 50,
					percentile: 50,
					summary: 'Analysis failed due to technical issues.',
				},
				summary: {
					title: 'Analysis Error',
					description: 'An error occurred while analyzing the contract.',
					points: [],
					riskAssessment: {
						level: 'neutral',
						label: 'Unknown',
						description: 'Analysis failed due to technical issues.',
					},
				},
				legalReferences: [],
				industryStandards: [],
				regulatoryGuidelines: [],
			};

			localStorage.setItem(
				`contractAnalysis_${docId}`,
				JSON.stringify(defaultAnalysis)
			);

			return defaultAnalysis;
		}
	} catch (error) {
		console.error('Error generating analysis:', error);
		throw error;
	}
}

// Function to fetch contract analysis data
export async function getContractAnalysis(analysisId: string) {
	try {
		// Check if we have cached analysis data
		const cachedAnalysis = localStorage.getItem(
			`contractAnalysis_${analysisId}`
		);
		console.log('cachedAnalysis', JSON.parse(cachedAnalysis));

		if (cachedAnalysis) {
			// Parse the cached data
			const parsedData = JSON.parse(cachedAnalysis);

			// Clean up any unexpected content
			return cleanAnalysisData(parsedData);
		}

		// If no cached data exists, try to regenerate the analysis
		console.log('No cached analysis found, regenerating...');
		
		// Call the FastAPI endpoint to regenerate the analysis
		const response = await fetch('http://localhost:8000/query', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				query: `Analyze this contract for sentiment, bias, and fairness. 
    Focus on identifying vendor-favorable vs customer-favorable clauses.
    Provide an overall fairness score out of 100, risk assessment, and recommendations.`,
				doc_id: analysisId,
			}),
		});

		if (!response.ok) {
			throw new Error(`Error regenerating analysis: ${response.statusText}`);
		}

		const responseData = await response.json();
		const parsedResponse = JSON.parse(responseData.response);
		
		// Store the regenerated analysis in localStorage
		localStorage.setItem(
			`contractAnalysis_${analysisId}`,
			JSON.stringify(parsedResponse)
		);

		return cleanAnalysisData(parsedResponse);
	} catch (error) {
		console.error('Error fetching analysis:', error);
		throw error;
	}
}

// Helper function to clean analysis data
function cleanAnalysisData(data: ContractAnalysis): ContractAnalysis {
	// Helper to clean text fields and remove JSON syntax
	const cleanText = (text: string): string => {
		if (!text) return '';

		// Remove JSON-like content
		return text
			.replace(/```json[\s\S]*?```/g, '')
			.replace(/```[\s\S]*?```/g, '')
			.replace(/```/g, '')
			.replace(/\{\s*"contractName"[\s\S]*?\}/g, '')
			.replace(/\{\s*"[\w]+"[\s\S]*?\}/g, '')
			.replace(/\\"/g, '"')
			.replace(/\\\\/g, '\\')
			.trim();
	};

	// Create a deep copy to avoid modifying the original
	const cleanData = { ...data };

	// Clean description
	if (typeof cleanData.description === 'string') {
		cleanData.description = cleanText(cleanData.description);
	}

	// Clean summary
	if (cleanData.summary) {
		if (typeof cleanData.summary.description === 'string') {
			cleanData.summary.description = cleanText(cleanData.summary.description);
		}

		if (cleanData.summary.points && Array.isArray(cleanData.summary.points)) {
			cleanData.summary.points = cleanData.summary.points.map(point => ({
				...point,
				title:
					typeof point.title === 'string'
						? cleanText(point.title)
						: point.title,
				description:
					typeof point.description === 'string'
						? cleanText(point.description)
						: point.description,
			}));
		}

		if (
			cleanData.summary.riskAssessment &&
			typeof cleanData.summary.riskAssessment.description === 'string'
		) {
			cleanData.summary.riskAssessment.description = cleanText(
				cleanData.summary.riskAssessment.description
			);
		}
	}

	// Clean notable clauses
	if (cleanData.notableClauses && Array.isArray(cleanData.notableClauses)) {
		cleanData.notableClauses = cleanData.notableClauses.map(clause => ({
			...clause,
			text:
				typeof clause.text === 'string'
					? cleanText(clause.text)
					: 'Contract clause text unavailable',
			analysis:
				typeof clause.analysis === 'string'
					? cleanText(clause.analysis)
					: clause.analysis,
			industryComparison:
				typeof clause.industryComparison === 'string'
					? cleanText(clause.industryComparison)
					: clause.industryComparison,
			recommendations: Array.isArray(clause.recommendations)
				? clause.recommendations.map(rec =>
						typeof rec === 'string' ? cleanText(rec) : rec
				  )
				: clause.recommendations,
		}));
	}

	// Clean industry benchmarking
	if (
		cleanData.industryBenchmarking &&
		typeof cleanData.industryBenchmarking.summary === 'string'
	) {
		cleanData.industryBenchmarking.summary = cleanText(
			cleanData.industryBenchmarking.summary
		);
	}

	return cleanData;
}
