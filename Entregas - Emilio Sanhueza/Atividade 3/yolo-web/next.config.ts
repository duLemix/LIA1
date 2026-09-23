import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The application runs inference entirely in the browser and does not need
  // a Node.js function. Exporting static files also avoids Vercel deriving a
  // Serverless Function name from the repository's space-containing path.
  output: "export",
};

export default nextConfig;
