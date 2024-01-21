import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/navbar"
const inter = Inter({ subsets: ["latin"] });
import Footer from "@/components/footer"
export const metadata: Metadata = {
  title: "Ayur Guru",
  description: "Your Ayurvedic Companion for a Balanced Life",
  icons: {
      icon: "/favicon.ico",
      apple: "/apple-icon.png"
  }
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="light">
      <body className={inter.className}>
      <Navbar />
      {children}
      <Footer />
      </body>
    </html>
  );
}
