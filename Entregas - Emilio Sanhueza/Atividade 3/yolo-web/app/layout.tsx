import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YONO — Browser-native ONNX inference",
  description: "Run object detection and image classification models locally in your browser.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
