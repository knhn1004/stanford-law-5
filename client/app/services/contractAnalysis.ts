// Mock data for contract analysis
const mockContractAnalysis = {
  id: 'mock-analysis-123',
  timestamp: new Date().toISOString(),
  contractName: 'Software License Agreement: SaaSCo - Enterprise Edition',
  description: 'This analysis examines the SaaSCo Enterprise License Agreement (v3.2) for sentiment, bias, and risk factors.',
  metrics: {
    overallFairnessScore: 68,
    potentialBiasIndicators: 12,
    highRiskClauses: 4,
    balancedClauses: 8
  },
  sentimentDistribution: {
    vendorFavorable: 45,
    balanced: 30,
    customerFavorable: 15,
    neutral: 10
  },
  notableClauses: [
    {
      type: 'Limitation of Liability',
      sentiment: 'biased-seller',
      sentimentLabel: 'Vendor-favorable',
      biasScore: 87,
      riskLevel: 'negative',
      riskLabel: 'High',
      text: 'IN NO EVENT SHALL VENDOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. VENDOR\'S TOTAL LIABILITY SHALL NOT EXCEED THE AMOUNTS PAID BY CUSTOMER IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.',
      analysis: 'This clause is heavily vendor-favorable with several concerning elements:',
      biasIndicators: [
        { label: 'Vendor Protection', value: 95 },
        { label: 'Customer Protection', value: 15 },
        { label: 'Negotiability', value: 22 }
      ],
      industryComparison: 'This limitation of liability clause is more restrictive than 78% of similar SaaS contracts in our database. Most enterprise SaaS agreements exclude liability for indirect damages but include exceptions for data breaches, IP infringement, and violations of confidentiality.',
      recommendations: [
        'Add exceptions for breach of confidentiality, data protection obligations, and intellectual property indemnification',
        'Increase liability cap to at least 24 months of fees',
        'Add mutual limitations on liability for both parties'
      ]
    },
    {
      type: 'Termination',
      sentiment: 'biased-seller',
      sentimentLabel: 'Vendor-favorable',
      biasScore: 72,
      riskLevel: 'negative',
      riskLabel: 'High',
      text: 'Vendor may terminate this Agreement immediately upon written notice if Customer: (a) fails to pay any fees when due; (b) breaches any term of this Agreement; or (c) becomes insolvent or files for bankruptcy. Customer may terminate this Agreement for material breach by Vendor only if Vendor fails to cure such breach within 30 days of written notice.',
      analysis: 'This clause demonstrates significant imbalance between the parties\' termination rights:',
      biasIndicators: [
        { label: 'Vendor Protection', value: 85 },
        { label: 'Customer Protection', value: 35 },
        { label: 'Negotiability', value: 40 }
      ],
      industryComparison: 'This termination clause is more restrictive than 65% of similar SaaS contracts in our database. Most enterprise SaaS agreements provide reciprocal termination rights with similar cure periods for both parties.',
      recommendations: [
        'Make cure periods reciprocal for both parties (30 days)',
        'Limit immediate termination to material breaches only',
        'For payment issues, require notice and a 15-day cure period',
        'Add customer right to terminate for convenience with 30-day notice'
      ]
    },
    {
      type: 'Indemnification',
      sentiment: 'biased-seller',
      sentimentLabel: 'Vendor-favorable',
      biasScore: 65,
      riskLevel: 'negative',
      riskLabel: 'Medium',
      text: 'Customer shall indemnify, defend, and hold harmless Vendor from and against any claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys\' fees) arising out of or relating to Customer\'s use of the Software or breach of this Agreement.',
      analysis: 'This indemnification clause is one-sided, requiring the customer to indemnify the vendor but not vice versa.',
      biasIndicators: [
        { label: 'Vendor Protection', value: 75 },
        { label: 'Customer Protection', value: 25 },
        { label: 'Negotiability', value: 50 }
      ],
      industryComparison: 'This indemnification clause is more restrictive than 60% of similar SaaS contracts in our database. Most enterprise SaaS agreements include mutual indemnification provisions.',
      recommendations: [
        'Add mutual indemnification for both parties',
        'Limit customer indemnification to claims arising from customer\'s breach',
        'Add vendor indemnification for claims related to the software\'s infringement of third-party intellectual property rights'
      ]
    },
    {
      type: 'Payment Terms',
      sentiment: 'neutral',
      sentimentLabel: 'Neutral',
      biasScore: 42,
      riskLevel: 'neutral',
      riskLabel: 'Low',
      text: 'Customer shall pay all fees in accordance with the pricing set forth in the Order Form. All payments are non-refundable and non-cancelable except as expressly set forth in this Agreement.',
      analysis: 'This payment terms clause is relatively balanced, with standard terms for SaaS agreements.',
      biasIndicators: [
        { label: 'Vendor Protection', value: 60 },
        { label: 'Customer Protection', value: 40 },
        { label: 'Negotiability', value: 55 }
      ],
      industryComparison: 'This payment terms clause is consistent with industry standards for SaaS agreements.',
      recommendations: [
        'Consider adding proration for unused portions of prepaid fees upon termination',
        'Add clarity on payment methods and currency'
      ]
    },
    {
      type: 'Data Protection',
      sentiment: 'biased-buyer',
      sentimentLabel: 'Customer-favorable',
      biasScore: 68,
      riskLevel: 'positive',
      riskLabel: 'Low',
      text: 'Vendor shall implement appropriate technical and organizational measures to protect Customer Data against unauthorized or unlawful processing, accidental loss, destruction, or damage. Vendor shall notify Customer without undue delay upon becoming aware of a Personal Data Breach.',
      analysis: 'This data protection clause provides strong protections for customer data.',
      biasIndicators: [
        { label: 'Vendor Protection', value: 30 },
        { label: 'Customer Protection', value: 70 },
        { label: 'Negotiability', value: 60 }
      ],
      industryComparison: 'This data protection clause is more protective than 75% of similar SaaS contracts in our database.',
      recommendations: [
        'Add specific timeframes for breach notification (e.g., within 72 hours)',
        'Include requirements for regular security audits'
      ]
    }
  ],
  industryBenchmarking: {
    fairnessScore: 68,
    percentile: 40,
    summary: 'This contract has a fairness score of 68/100, which places it in the bottom 40% of analyzed SaaS agreements in the enterprise software sector. The most significant deviations from industry norms are in the limitation of liability, termination rights, and service level agreement clauses.'
  },
  summary: {
    title: 'Summary and Recommendations',
    description: 'This Software License Agreement significantly favors the vendor across multiple key provisions. The most concerning areas are:',
    points: [
      {
        title: 'Limitation of Liability',
        description: 'Extremely vendor-favorable with a low cap and broad exclusions'
      },
      {
        title: 'Termination Rights',
        description: 'Unbalanced in favor of the vendor with immediate termination rights'
      },
      {
        title: 'Indemnification',
        description: 'One-sided protection for the vendor with limited reciprocal coverage'
      },
      {
        title: 'Warranty',
        description: 'Minimal warranties with extensive disclaimers'
      }
    ],
    riskAssessment: {
      level: 'negative',
      label: 'High Risk',
      description: 'This contract presents High Risk due to the imbalanced terms and significant customer exposure. We recommend prioritizing negotiation of the highlighted clauses to achieve more balanced terms.'
    }
  }
};

// Function to fetch contract analysis data
export async function getContractAnalysis(analysisId: string) {
  // In a real implementation, this would fetch data from the FastAPI backend
  // For now, we'll just return the mock data after a delay to simulate network latency
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(mockContractAnalysis);
    }, 1000);
  });
} 