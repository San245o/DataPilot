import type { Metadata } from "next";
import { HistoryView } from "@/components/home/history-view";

export const metadata: Metadata = {
  title: "History — DataPilot",
  description: "Browse and reopen your past DataPilot analyses.",
};

export default function HistoryPage() {
  return <HistoryView />;
}
