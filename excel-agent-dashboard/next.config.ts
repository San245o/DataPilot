import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  experimental: {
    proxyClientMaxBodySize: "10mb",
  },
  async rewrites() {
    const backendBaseUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backendBaseUrl}/:path*`,
      },
    ]
  },
};

export default nextConfig;
