"use client";

import Link from "next/link";
import { ArrowLeft, FileSpreadsheet } from "lucide-react";
import "./home.css";

const HISTORY = [
  { name: "Sales Performance", file: "sales_2026.xlsx", when: "Today" },
  { name: "Student Performance", file: "student_results.xlsx", when: "Yesterday" },
  { name: "Product Price Analysis", file: "product_prices.csv", when: "3 days ago" },
  { name: "Marketing Funnel", file: "funnel_q2.xlsx", when: "Last week" },
  { name: "Inventory Snapshot", file: "inventory.csv", when: "2 weeks ago" },
];

export function HistoryView() {
  return (
    <div className="datapilot-home relative min-h-screen px-5 py-12 sm:px-8">
      <div className="bg-grid-texture pointer-events-none absolute inset-x-0 top-0 h-72 [mask-image:linear-gradient(to_bottom,black,transparent)]" />
      <div className="relative mx-auto w-full max-w-3xl">
        <Link
          href="/home"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-primary"
        >
          <ArrowLeft className="size-4" />
          Back to Home
        </Link>
        <h1 className="mt-4 text-2xl font-semibold tracking-tight text-foreground">
          Analysis history
        </h1>
        <p className="mt-1.5 text-sm text-muted-foreground">
          Reopen any of your past analyses.
        </p>
        <div className="mt-8 divide-y divide-border rounded-2xl border border-border bg-card shadow-card">
          {HISTORY.map((item) => (
            <Link
              key={item.name}
              href="/dashboard"
              className="flex items-center gap-4 px-5 py-4 transition-colors hover:bg-surface-hover"
            >
              <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-emerald-soft text-primary ring-1 ring-primary/15">
                <FileSpreadsheet className="size-4.5" strokeWidth={2} />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-foreground">
                  {item.name}
                </span>
                <span className="block truncate text-xs text-muted-foreground">
                  {item.file}
                </span>
              </span>
              <span className="shrink-0 text-xs text-muted-foreground">
                {item.when}
              </span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
