import type { Metadata } from "next";
import { HomeView } from "@/components/home/home-view";

export const metadata: Metadata = {
  title: "Home — DataPilot",
  description:
    "Your DataPilot launchpad. Start a new analysis, continue recent work, and learn how to turn spreadsheets into insights.",
};

export default function HomePage() {
  return <HomeView />;
}
