'use client';

import React, { useEffect, useState } from 'react';
import SentimentBadge from '../components/SentimentBadge';
import ProgressBar from '../components/ProgressBar';
import {
	getContractAnalysis,
	ContractAnalysis,
} from '../services/contractAnalysis';
import Link from 'next/link';

export default function ContractSentimentPage() {
	const [analysis, setAnalysis] = useState<ContractAnalysis | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	// Helper function to clean text content from JSON and markdown artifacts
	const cleanTextContent = (text: string): string => {
		if (!text) return '';

		return text
			.replace(/```json[\s\S]*?```/g, '')
			.replace(/```[\s\S]*?```/g, '')
			.replace(/```/g, '')
			.replace(/\{\s*"contractName"[\s\S]*?\}/g, '')
			.replace(/^\s*```\s*|\s*```\s*$/g, '')
			.replace(/^```\s*json/i, '')
			.replace(/\\"/g, '"')
			.replace(/\\\\/g, '\\')
			.replace(/^[`]+|[`]+$/g, '')
			.trim();
	};

	useEffect(() => {
		const fetchAnalysis = async () => {
			try {
				// Get the analysis ID from localStorage
				const analysisId = localStorage.getItem('contractAnalysisId');

				if (!analysisId) {
					setError('No analysis ID found. Please upload a contract first.');
					setLoading(false);
					return;
				}

				// Fetch the analysis data
				const data = await getContractAnalysis(analysisId);

				// The service should now return cleaned data, but we'll still do a final check here
				// for any JSON that might have slipped through

				// Clean description for display
				if (typeof data.description === 'string') {
					data.description = cleanTextContent(data.description);
				}

				// Fix contract name formatting
				if (data.contractName.startsWith('"')) {
					data.contractName = data.contractName.replace(/^"|"$/g, '');
				}

				// Clean summary content
				if (data.summary && typeof data.summary.description === 'string') {
					data.summary.description = cleanTextContent(data.summary.description);
				}

				if (
					data.summary &&
					data.summary.points &&
					Array.isArray(data.summary.points)
				) {
					data.summary.points = data.summary.points.map(point => ({
						...point,
						description:
							typeof point.description === 'string'
								? cleanTextContent(point.description)
								: point.description,
					}));
				}

				// Clean notable clause text
				if (data.notableClauses && Array.isArray(data.notableClauses)) {
					data.notableClauses = data.notableClauses.map(clause => ({
						...clause,
						text:
							typeof clause.text === 'string'
								? cleanTextContent(clause.text)
								: 'Contract clause text unavailable',
					}));
				}

				setAnalysis(data as ContractAnalysis);
			} catch (err) {
				setError('Failed to load contract analysis. Please try again.');
				console.error('Error fetching analysis:', err);
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
					{[1, 2, 3, 4].map(i => (
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
							{[1, 2, 3, 4].map(i => (
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
	const formattedDate = new Date(analysis.timestamp).toLocaleString('en-US', {
		year: 'numeric',
		month: 'long',
		day: 'numeric',
		hour: '2-digit',
		minute: '2-digit',
		timeZoneName: 'short',
	});

	// Format contract name to be more readable
	const formattedContractName = analysis.contractName
		.replace(/([A-Z])/g, ' $1') // Add space before capital letters
		.replace(/_/g, ' ') // Replace underscores with spaces
		.replace(/-/g, '-') // Keep hyphens
		.replace(/\s+/g, ' ') // Replace multiple spaces with single space
		.trim();

	return (
		<div className="container mx-auto px-4 py-8 max-w-7xl">
			<div className="flex justify-between items-center mb-8">
				<div className="text-2xl font-bold text-blue-600">ContractSentinel</div>
				<div className="text-sm text-gray-500">Analyzed: {formattedDate}</div>
			</div>

			<div className="bg-white rounded-lg shadow-md p-6 mb-6">
				<h1 className="text-2xl font-bold mb-2">{formattedContractName}</h1>
				<div className="text-gray-600 whitespace-pre-line">
					{analysis.description.startsWith('```json{') || analysis.description.includes('json {') ? (
						<p>Contract analysis completed successfully.</p>
					) : analysis.description && analysis.description.includes('**') ? (
						analysis.description
							.split('\n')
							.map((line, i) => (
								<React.Fragment key={i}>
									{line.trim() && (
										<>
											{line.startsWith('**') && line.endsWith('**') ? (
												<strong className="font-bold block mt-2">
													{line.replace(/\*\*/g, '')}
												</strong>
											) : (
												<span>{line}</span>
											)}
											<br />
										</>
									)}
								</React.Fragment>
							))
					) : (
						<p>{analysis.description}</p>
					)}
				</div>
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
					{/* Chart visualization - in a real app you would use a chart library like Chart.js, Recharts, etc. */}
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
							{analysis.notableClauses && analysis.notableClauses.length > 0 ? (
								analysis.notableClauses.map((clause, index) => (
									<tr
										key={index}
										className={
											index < analysis.notableClauses.length - 1
												? 'border-b border-gray-100'
												: ''
										}
									>
										<td className="py-3 px-4">{clause.type}</td>
										<td className="py-3 px-4">
											<SentimentBadge
												type={clause.sentiment}
												label={clause.sentimentLabel}
											/>
										</td>
										<td className="py-3 px-4">
											<ProgressBar value={clause.biasScore} color="bg-red-500" />
										</td>
										<td className="py-3 px-4">
											<SentimentBadge
												type={clause.riskLevel}
												label={clause.riskLabel}
											/>
										</td>
									</tr>
								))
							) : (
								<tr>
									<td colSpan={4} className="py-6 text-center text-gray-500">
										No notable clauses found in analysis.
									</td>
								</tr>
							)}
						</tbody>
					</table>
				</div>
			</div>

			<div className="bg-white rounded-lg shadow-md p-6 mb-8">
				<h2 className="text-xl font-semibold mb-4 pb-2 border-b border-gray-200">
					Detailed Clause Analysis
				</h2>

				{analysis.notableClauses && analysis.notableClauses.length > 0 ? (
					analysis.notableClauses.map((clause, i) => (
						<div
							key={i}
							className="bg-white rounded-lg shadow-md p-6 mb-4 border-l-4 border-blue-600"
						>
							<div className="flex justify-between items-start mb-4">
								<div>
									<h3 className="text-lg font-semibold">
										{clause.type || 'Clause ' + (i + 1)}
									</h3>
									<SentimentBadge
										type={clause.sentiment}
										label={clause.sentimentLabel || clause.sentiment}
									/>
								</div>
								<div className="text-right">
									<div
										className={`text-sm font-medium px-3 py-1 rounded ${
											clause.biasScore >= 8
												? 'bg-green-100 text-green-800'
												: clause.biasScore >= 5
												? 'bg-yellow-100 text-yellow-800'
												: 'bg-red-100 text-red-800'
										}`}
									>
										Fairness: {clause.biasScore}/10
									</div>
									<div className="text-xs text-gray-500 mt-1">
										Section {i + 1}
									</div>
								</div>
							</div>

							<div className="mb-4 p-4 bg-gray-50 rounded border border-gray-200 whitespace-pre-wrap text-sm">
								<div className="font-medium mb-1">Text:</div>
								{clause.text}
							</div>

							<div className="flex flex-col md:flex-row gap-6">
								<div className="md:w-1/3">
									<div className="font-medium mb-2">Sentiment Indicators:</div>
									{clause.biasIndicators && clause.biasIndicators.length > 0 ? (
										clause.biasIndicators.map((indicator, i) => (
											<div key={i} className="mb-2">
												<ProgressBar
													value={indicator.value}
													color={
														i === 0
															? 'bg-red-500'
															: i === 1
															? 'bg-blue-500'
															: 'bg-gray-500'
													}
													label={indicator.label}
												/>
											</div>
										))
									) : (
										<div className="text-sm text-gray-500">No sentiment indicators</div>
									)}
								</div>

								<div className="flex-1">
									<div className="font-medium mb-2">Industry Comparison:</div>
									<p className="text-sm">{clause.industryComparison}</p>
								</div>
							</div>

							<div className="recommendation-item mt-4">
								<div className="font-medium mb-2">Recommended Changes:</div>
								<ol className="list-decimal pl-5">
									{clause.recommendations && clause.recommendations.length > 0 ? (
										clause.recommendations.map((rec, i) => (
											<li key={i} className="mb-1">
												{rec}
											</li>
										))
									) : (
										<li className="text-gray-500">No recommendations available</li>
									)}
								</ol>
							</div>
						</div>
					))
				) : (
					<div className="py-6 text-center text-gray-500">
						No detailed clause analysis available.
					</div>
				)}
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
					{analysis.summary?.title || 'Analysis Summary'}
				</h2>
				<div className="mb-4 whitespace-pre-line">
					{analysis.summary?.description?.startsWith('```') || 
					 analysis.summary?.description?.includes('{"contractName"') || 
					 analysis.summary?.description?.includes('json {') ? (
						<p>This contract has been analyzed for sentiment, bias, and fairness.</p>
					) : (
						analysis.summary?.description || 'No summary description available'
					)}
				</div>

				{analysis.summary?.points && analysis.summary.points.length > 0 ? (
					<ol className="list-decimal pl-5 mb-6">
						{analysis.summary.points.map((point, index) => (
							<li key={index} className="mb-2">
								<strong>{point.title || 'Point'}</strong>:{' '}
								{point.description && !point.description.includes('`') 
									? point.description
									: 'See full analysis for details.'}
							</li>
						))}
					</ol>
				) : (
					<p className="mb-6">See above analysis for key contract points.</p>
				)}

				<div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
					<strong className="text-lg">Overall Risk Assessment:</strong>{' '}
					<span
						className={`sentiment-badge ${
							analysis.summary?.riskAssessment?.level === 'negative'
								? 'sentiment-negative'
								: analysis.summary?.riskAssessment?.level === 'positive'
								? 'sentiment-positive'
								: 'sentiment-neutral'
						}`}
					>
						{analysis.summary?.riskAssessment?.label || 'N/A'}
					</span>
					<p className="mt-2">{analysis.summary?.riskAssessment?.description || 'No risk assessment available'}</p>
				</div>
			</div>
		</div>
	);
}
