import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Clause Clarity - AI-Powered Contract Analysis",
  description: "Clause Clarity helps you analyze contracts for bias, fairness, and sentiment using advanced AI technology.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
        suppressHydrationWarning
      >
        <header className="border-b border-slate-200 bg-white">
          <div className="container mx-auto px-4 py-4 max-w-7xl">
            <div className="flex justify-between items-center">
              <Link href="/">
                <div className="flex items-center space-x-2">
                  <svg className="w-8 h-8 text-slate-700" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M7 9H17M7 13H17M9 17H15M9.2 4H14.8C15.9201 4 16.4802 4 16.908 4.21799C17.2843 4.40973 17.5903 4.71569 17.782 5.09202C18 5.51984 18 6.0799 18 7.2V16.8C18 17.9201 18 18.4802 17.782 18.908C17.5903 19.2843 17.2843 19.5903 16.908 19.782C16.4802 20 15.9201 20 14.8 20H9.2C8.0799 20 7.51984 20 7.09202 19.782C6.71569 19.5903 6.40973 19.2843 6.21799 18.908C6 18.4802 6 17.9201 6 16.8V7.2C6 6.0799 6 5.51984 6.21799 5.09202C6.40973 4.71569 6.71569 4.40973 7.09202 4.21799C7.51984 4 8.0799 4 9.2 4Z" 
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span className="text-xl font-serif font-bold text-slate-800">Clause Clarity</span>
                </div>
              </Link>
              <nav>
                <ul className="flex space-x-8">
                  <li>
                    <Link href="/contract-sentiment" className="text-slate-600 hover:text-slate-800 font-medium">
					Document Analysis
                    </Link>
                  </li>
                </ul>
              </nav>
            </div>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
