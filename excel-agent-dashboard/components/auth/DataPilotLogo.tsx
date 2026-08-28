"use client";

import { Sparkles } from "lucide-react";

export function DataPilotLogo({ tone = "light" }: { tone?: "light" | "dark" }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[var(--auth-accent)] text-[var(--auth-accent-fg)] shadow-[var(--auth-shadow-accent)]">
        <Sparkles className="h-[18px] w-[18px]" />
      </span>
      <span
        className={`text-lg font-semibold tracking-tight ${
          tone === "light" ? "text-[var(--auth-panel-fg)]" : "text-foreground"
        }`}
      >
        DataPilot
      </span>
    </div>
  );
}

export default DataPilotLogo;
