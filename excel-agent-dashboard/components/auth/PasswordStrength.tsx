"use client";

export function scorePassword(value: string): number {
  if (!value) return 0;
  let score = 0;
  if (value.length >= 8) score++;
  if (/[A-Z]/.test(value) && /[a-z]/.test(value)) score++;
  if (/\d/.test(value)) score++;
  if (/[^A-Za-z0-9]/.test(value)) score++;
  return score; // 0-4
}

const LABELS = ["", "Weak", "Fair", "Good", "Strong"];
const COLORS = [
  "",
  "bg-red-500",
  "bg-amber-500",
  "bg-teal-500",
  "bg-emerald-500",
];

export function PasswordStrength({ value }: { value: string }) {
  const score = scorePassword(value);
  if (!value) return null;

  return (
    <div className="space-y-1.5 pt-0.5">
      <div className="flex gap-1.5">
        {[1, 2, 3, 4].map((i) => (
          <span
            key={i}
            className={`h-1.5 flex-1 rounded-full transition-colors ${
              i <= score ? COLORS[score] : "bg-slate-200"
            }`}
          />
        ))}
      </div>
      <p className="text-xs text-slate-500">
        Password strength: <span className="font-medium text-slate-900">{LABELS[score]}</span>
      </p>
    </div>
  );
}

export default PasswordStrength;
