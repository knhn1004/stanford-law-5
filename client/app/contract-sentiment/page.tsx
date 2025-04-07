'use client';

import React, { useEffect, useState } from 'react';
import SentimentBadge, { SentimentType } from '../components/SentimentBadge';
import {
	getContractAnalysis,
	ContractAnalysis,
} from '../services/contractAnalysis';
import Link from 'next/link';
import {
	PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar,
	XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';
import ProgressBar from '../components/ProgressBar';

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
			<div className="container mx-auto px-4 py-12 max-w-7xl">
				<div className="flex justify-between items-center mb-8">
					<div className="text-2xl font-serif font-bold text-slate-700">
						Document Analysis
					</div>
					<div className="text-sm text-slate-500">Processing...</div>
				</div>

				<div className="legal-card mb-8">
					<div className="animate-pulse">
						<div className="h-8 bg-slate-100 rounded w-3/4 mb-4"></div>
						<div className="h-4 bg-slate-100 rounded w-full mb-2"></div>
						<div className="h-4 bg-slate-100 rounded w-5/6"></div>
					</div>
				</div>

				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
					{[1, 2, 3, 4].map(i => (
						<div key={i} className="legal-card">
							<div className="animate-pulse">
								<div className="h-8 bg-slate-100 rounded w-16 mx-auto mb-2"></div>
								<div className="h-4 bg-slate-100 rounded w-24 mx-auto"></div>
							</div>
						</div>
					))}
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="container mx-auto px-4 py-12 max-w-7xl">
				<div className="flex justify-between items-center mb-8">
					<div className="text-2xl font-serif font-bold text-slate-700">
						Document Analysis
					</div>
				</div>

				<div className="legal-card text-center py-12">
					<div className="text-rose-700 text-xl font-serif mb-4">Error</div>
					<p className="text-slate-600 mb-6">{error}</p>
					<Link
						href="/"
						className="legal-button inline-block hover:bg-slate-100 transition-colors"
					>
						Return to Upload
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
		<div className="container mx-auto px-4 py-12 max-w-7xl">
			<div className="flex justify-between items-center mb-8">
				<div className="text-2xl font-serif font-bold text-slate-700">
					Document Analysis
				</div>
				<div className="text-sm text-slate-500 font-medium">
					Analyzed: {formattedDate}
				</div>
			</div>

			<div className="legal-card mb-8">
				<h1 className="text-2xl font-serif font-bold text-slate-800 mb-4">
					{formattedContractName}
				</h1>
				<div className="text-slate-600 whitespace-pre-line leading-relaxed">
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
												<strong className="font-serif block mt-4 mb-2 text-slate-800">
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

			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
				<div className="legal-card text-center">
					<div className="text-3xl font-serif font-bold text-slate-700 mb-2">
						{analysis.metrics.overallFairnessScore}
					</div>
					<div className="text-sm font-medium text-slate-600">Fairness Score</div>
				</div>
				<div className="legal-card text-center">
					<div className="text-3xl font-serif font-bold text-amber-700 mb-2">
						{analysis.metrics.potentialBiasIndicators}
					</div>
					<div className="text-sm font-medium text-slate-600">Risk Indicators</div>
				</div>
				<div className="legal-card text-center">
					<div className="text-3xl font-serif font-bold text-rose-700 mb-2">
						{analysis.metrics.highRiskClauses}
					</div>
					<div className="text-sm font-medium text-slate-600">Critical Clauses</div>
				</div>
				<div className="legal-card text-center">
					<div className="text-3xl font-serif font-bold text-emerald-700 mb-2">
						{analysis.metrics.balancedClauses}
					</div>
					<div className="text-sm font-medium text-slate-600">Balanced Clauses</div>
				</div>
			</div>

			<div className="legal-card mb-12">
				<h2 className="text-xl font-serif font-bold text-slate-800 mb-6 pb-2 border-b border-slate-200">
					Clause Distribution Analysis
				</h2>
				<div className="h-80 mb-8">
					<ResponsiveContainer width="100%" height="100%">
						<PieChart>
							<Pie
								data={[
									{ name: 'Vendor-favorable', value: analysis.sentimentDistribution.vendorFavorable },
									{ name: 'Balanced', value: analysis.sentimentDistribution.balanced },
									{ name: 'Customer-favorable', value: analysis.sentimentDistribution.customerFavorable },
									{ name: 'Neutral', value: analysis.sentimentDistribution.neutral }
								]}
								cx="50%"
								cy="50%"
								labelLine={true}
								outerRadius={120}
								fill="#8884d8"
								dataKey="value"
								label={({
									cx,
									cy,
									midAngle,
									outerRadius,
									value,
									name
								}) => {
									const RADIAN = Math.PI / 180;
									const radius = outerRadius * 1.2;
									const x = cx + radius * Math.cos(-midAngle * RADIAN);
									const y = cy + radius * Math.sin(-midAngle * RADIAN);
									return (
										<text
											x={x}
											y={y}
											fill="var(--slate-600)"
											textAnchor={x > cx ? 'start' : 'end'}
											dominantBaseline="central"
											className="text-sm"
										>
											{`${name} (${value}%)`}
										</text>
									);
								}}
							>
								<Cell fill="#be123c" />
								<Cell fill="#047857" />
								<Cell fill="#334155" />
								<Cell fill="#64748b" />
							</Pie>
							<Tooltip />
						</PieChart>
					</ResponsiveContainer>
				</div>
				<div className="flex flex-wrap justify-center gap-6">
					<div className="flex items-center gap-2">
						<div className="w-3 h-3 rounded-full bg-rose-700"></div>
						<span className="text-sm font-medium text-slate-600">
							Vendor-favorable ({analysis.sentimentDistribution.vendorFavorable}%)
						</span>
					</div>
					<div className="flex items-center gap-2">
						<div className="w-3 h-3 rounded-full bg-emerald-700"></div>
						<span className="text-sm font-medium text-slate-600">
							Balanced ({analysis.sentimentDistribution.balanced}%)
						</span>
					</div>
					<div className="flex items-center gap-2">
						<div className="w-3 h-3 rounded-full bg-slate-700"></div>
						<span className="text-sm font-medium text-slate-600">
							Customer-favorable ({analysis.sentimentDistribution.customerFavorable}%)
						</span>
					</div>
					<div className="flex items-center gap-2">
						<div className="w-3 h-3 rounded-full bg-slate-500"></div>
						<span className="text-sm font-medium text-slate-600">
							Neutral ({analysis.sentimentDistribution.neutral}%)
						</span>
					</div>
				</div>
			</div>

			<div className="legal-card mb-12">
				<h2 className="text-xl font-serif font-bold text-slate-800 mb-6 pb-2 border-b border-slate-200">
					Notable Clauses Analysis
				</h2>
				<div className="overflow-x-auto">
					<table className="w-full">
						<thead>
							<tr className="border-b border-slate-200">
								<th className="py-3 px-4 text-left text-sm font-serif font-medium text-slate-600">
									Clause Type
								</th>
								<th className="py-3 px-4 text-left text-sm font-serif font-medium text-slate-600">
									Sentiment
								</th>
								<th className="py-3 px-4 text-left text-sm font-serif font-medium text-slate-600">
									Risk Score
								</th>
								<th className="py-3 px-4 text-left text-sm font-serif font-medium text-slate-600">
									Risk Level
								</th>
							</tr>
						</thead>
						<tbody>
							{analysis.notableClauses && analysis.notableClauses.length > 0 ? (
								analysis.notableClauses.map((clause, index) => (
									<tr
										key={index}
										className={index < analysis.notableClauses.length - 1 ? 'border-b border-slate-100' : ''}
									>
										<td className="py-4 px-4 font-medium text-slate-800">{clause.type}</td>
										<td className="py-4 px-4">
											<SentimentBadge type={clause.sentiment.toLowerCase() as SentimentType} label={clause.sentimentLabel} />
										</td>
										<td className="py-4 px-4">
											<ProgressBar value={clause.biasScore} color="bg-slate-700" />
										</td>
										<td className="py-4 px-4">
											<SentimentBadge type={clause.riskLevel.toLowerCase() as SentimentType} label={clause.riskLabel} />
										</td>
									</tr>
								))
							) : (
								<tr>
									<td colSpan={4} className="py-8 text-center text-slate-500">
										No notable clauses found in analysis.
									</td>
								</tr>
							)}
						</tbody>
					</table>
				</div>
			</div>

			<div className="legal-card mb-12">
				<h2 className="text-xl font-serif font-bold text-slate-800 mb-6 pb-2 border-b border-slate-200">
					Detailed Analysis
				</h2>

				<div className="space-y-6">
					{analysis.notableClauses.map((clause, index) => (
						<div key={index} className="border-b border-slate-200 last:border-0 pb-6 last:pb-0">
							<div className="flex items-center justify-between mb-4">
								<h4 className="font-medium text-slate-800">
									{clause.type}
								</h4>
								<div className="flex items-center space-x-2">
									<span className={`px-3 py-1 rounded-full text-sm ${
										clause.riskLevel.toLowerCase() === 'high'
											? 'bg-red-100 text-red-800'
											: clause.riskLevel.toLowerCase() === 'medium'
											? 'bg-yellow-100 text-yellow-800'
											: 'bg-green-100 text-green-800'
									}`}>
										{clause.riskLabel}
									</span>
									<span className={`px-3 py-1 rounded-full text-sm ${
										clause.sentiment.toLowerCase() === 'negative'
											? 'bg-red-100 text-red-800'
											: clause.sentiment.toLowerCase() === 'neutral'
											? 'bg-slate-100 text-slate-800'
											: 'bg-green-100 text-green-800'
									}`}>
										{clause.sentimentLabel}
									</span>
								</div>
							</div>
							<div className="bg-slate-50 p-4 rounded-lg mb-4">
								<p className="font-mono text-sm text-slate-700">
									{clause.text}
								</p>
							</div>
							<p className="text-slate-600 mb-4">{clause.analysis}</p>
							
							{/* Legal Precedents */}
							{clause.legalPrecedents && clause.legalPrecedents.length > 0 && (
								<div className="mb-4">
									<h5 className="font-medium text-slate-700 mb-2">Legal Precedents</h5>
									<div className="space-y-2">
										{clause.legalPrecedents.map((precedent, idx) => (
											<div key={idx} className="bg-blue-50 p-3 rounded">
												<div className="font-medium text-blue-800">{precedent.case}</div>
												<div className="text-sm text-blue-600">{precedent.relevance}</div>
												<div className="text-sm text-slate-600">{precedent.implication}</div>
											</div>
										))}
									</div>
								</div>
							)}

							{/* Bias indicators */}
							<div className="mb-4">
								<h5 className="font-medium text-slate-700 mb-2">
									Bias Indicators
								</h5>
								<div className="space-y-2">
									{clause.biasIndicators.map((indicator, idx) => (
										<div key={idx} className="flex items-center justify-between">
											<span className="text-sm text-slate-600">
												{indicator.label}
											</span>
											<ProgressBar
												value={indicator.value}
												color={idx === 0 ? 'bg-rose-700' : idx === 1 ? 'bg-slate-700' : 'bg-slate-500'}
												label={indicator.label}
											/>
										</div>
									))}
								</div>
							</div>

							{/* Industry comparison */}
							<div className="mb-4">
								<h5 className="font-medium text-slate-700 mb-2">
									Industry Comparison
								</h5>
								<p className="text-sm text-slate-600">
									{clause.industryComparison}
								</p>
							</div>

							{/* Recommendations */}
							<div>
								<h5 className="font-medium text-slate-700 mb-2">
									Recommendations
								</h5>
								<ul className="list-disc list-inside space-y-1">
									{clause.recommendations.map((rec, idx) => (
										<li key={idx} className="text-sm text-slate-600">
											{rec}
										</li>
									))}
								</ul>
							</div>
						</div>
					))}
				</div>
			</div>

			<div className="legal-card mb-12">
				<h2 className="text-xl font-serif font-bold text-slate-800 mb-6 pb-2 border-b border-slate-200">
					Industry Benchmarking
				</h2>
				<div className="h-96 mb-8">
					<ResponsiveContainer width="100%" height="100%">
						<BarChart
							data={[
								{
									category: 'Overall Fairness',
									current: analysis.metrics.overallFairnessScore,
									industry: 5
								},
								{
									category: 'Risk Indicators',
									current: analysis.metrics.potentialBiasIndicators,
									industry: 8
								},
								{
									category: 'Critical Clauses',
									current: analysis.metrics.highRiskClauses,
									industry: 4
								},
								{
									category: 'Balanced Clauses',
									current: analysis.metrics.balancedClauses,
									industry: 3
								}
							]}
							margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
						>
							<CartesianGrid strokeDasharray="3 3" stroke="#e2e2e2" />
							<XAxis dataKey="category" tick={{ fill: '#4a4a4a' }} />
							<YAxis tick={{ fill: '#4a4a4a' }} />
							<Tooltip />
							<Legend />
							<Bar dataKey="current" name="This Document" fill="#334155" />
							<Bar dataKey="industry" name="Industry Average" fill="#64748b" />
						</BarChart>
					</ResponsiveContainer>
				</div>
				<p className="text-slate-600 leading-relaxed">{analysis.industryBenchmarking.summary}</p>
			</div>

			<div className="legal-card">
				<h2 className="text-xl font-serif font-bold text-slate-800 mb-6 pb-2 border-b border-slate-200">
					Executive Summary
				</h2>
				<div className="mb-6 text-slate-600 leading-relaxed whitespace-pre-line">
					{analysis.summary?.description?.startsWith('```') || 
					 analysis.summary?.description?.includes('{"contractName"') || 
					 analysis.summary?.description?.includes('json {') ? (
						<p>This document has been analyzed for legal compliance, risk factors, and fairness metrics.</p>
					) : (
						analysis.summary?.description || 'No summary description available'
					)}
				</div>

				{analysis.summary?.points && analysis.summary.points.length > 0 && (
					<div className="mb-8">
						<h3 className="font-serif font-medium text-slate-800 mb-4">Key Findings</h3>
						<ul className="space-y-3">
							{analysis.summary.points.map((point, index) => (
								<li key={index} className="flex gap-3">
									<div className="flex-shrink-0 w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center">
										<span className="text-sm font-medium text-slate-700">{index + 1}</span>
									</div>
									<div>
										<strong className="font-medium text-slate-800">{point.title || 'Finding'}</strong>:{' '}
										{point.description && !point.description.includes('`') 
											? point.description
											: 'See full analysis for details.'}
									</div>
								</li>
							))}
						</ul>
					</div>
				)}

				<div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
					<h3 className="font-serif font-bold text-slate-800 mb-3">Overall Risk Assessment</h3>
					<div className="flex items-center gap-3 mb-4">
						<SentimentBadge
							type={(analysis.summary?.riskAssessment?.level?.toLowerCase() as SentimentType) || "neutral"}
							label={analysis.summary?.riskAssessment?.label || 'Not Available'}
						/>
					</div>
					<p className="text-slate-600 leading-relaxed">
						{analysis.summary?.riskAssessment?.description || 'No risk assessment available'}
					</p>
				</div>
			</div>

			{/* Legal References Section */}
			{analysis.legalReferences && analysis.legalReferences.length > 0 && (
				<div className="legal-card mb-8">
					<h3 className="text-lg font-serif font-bold text-slate-700 mb-4">
						Legal References
					</h3>
					<div className="space-y-4">
						{analysis.legalReferences.map((ref, index) => (
							<div key={index} className="border-l-4 border-blue-500 pl-4 py-3 hover:bg-slate-50 transition-colors">
								<a 
									href={ref.url} 
									target="_blank" 
									rel="noopener noreferrer" 
									className="block group"
								>
									<h4 className="font-medium text-blue-600 group-hover:text-blue-800 transition-colors">
										{ref.title}
									</h4>
									<p className="text-sm text-slate-600 mt-1 mb-2">
										{ref.description}
									</p>
									<div className="flex items-center text-sm text-blue-500 group-hover:text-blue-700">
										<span>Read more</span>
										<svg 
											className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" 
											fill="none" 
											stroke="currentColor" 
											viewBox="0 0 24 24"
										>
											<path 
												strokeLinecap="round" 
												strokeLinejoin="round" 
												strokeWidth={2} 
												d="M9 5l7 7-7 7" 
											/>
										</svg>
									</div>
								</a>
							</div>
						))}
					</div>
				</div>
			)}

			{/* Industry Standards Section */}
			{analysis.industryStandards && analysis.industryStandards.length > 0 && (
				<div className="legal-card mb-8">
					<h3 className="text-lg font-serif font-bold text-slate-700 mb-4">
						Industry Standards
					</h3>
					<div className="space-y-4">
						{analysis.industryStandards.map((standard, index) => (
							<div key={index} className="flex items-start space-x-4">
								<div className={`px-3 py-1 rounded-full text-sm ${
									standard.complianceStatus === 'Compliant' 
										? 'bg-green-100 text-green-800'
										: standard.complianceStatus === 'Non-Compliant'
										? 'bg-red-100 text-red-800'
										: 'bg-yellow-100 text-yellow-800'
								}`}>
									{standard.complianceStatus}
								</div>
								<div>
									<h4 className="font-medium text-slate-800">
										{standard.name}
									</h4>
									<p className="text-slate-600 text-sm">
										{standard.description}
									</p>
								</div>
							</div>
						))}
					</div>
				</div>
			)}

			{/* Regulatory Guidelines Section */}
			{analysis.regulatoryGuidelines && analysis.regulatoryGuidelines.length > 0 && (
				<div className="legal-card mb-8">
					<h3 className="text-lg font-serif font-bold text-slate-700 mb-4">
						Regulatory Guidelines
					</h3>
					<div className="space-y-4">
						{analysis.regulatoryGuidelines.map((guideline, index) => (
							<div key={index} className="border-l-4 border-purple-500 pl-4">
								<h4 className="font-medium text-slate-800">
									{guideline.regulation}
								</h4>
								<p className="text-slate-600 text-sm mb-2">
									{guideline.relevance}
								</p>
								<div className={`inline-block px-3 py-1 rounded-full text-sm ${
									guideline.complianceStatus === 'Compliant'
										? 'bg-green-100 text-green-800'
										: guideline.complianceStatus === 'Non-Compliant'
										? 'bg-red-100 text-red-800'
										: 'bg-yellow-100 text-yellow-800'
								}`}>
									{guideline.complianceStatus}
								</div>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
}
