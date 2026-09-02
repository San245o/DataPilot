import type { Metadata } from "next";
import { TutorialsView } from "@/components/home/tutorials-view";

export const metadata: Metadata = {
  title: "Tutorials — DataPilot",
  description:
    "Video tutorials to help you get more from your spreadsheets with DataPilot.",
};

export default function TutorialsPage() {
  return <TutorialsView />;
}
