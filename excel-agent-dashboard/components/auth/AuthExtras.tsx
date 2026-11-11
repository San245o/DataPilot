"use client";

import type {
  ButtonHTMLAttributes,
  ReactNode,
} from "react";

export function AuthHeading({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div className="mb-6">
      <h1 className="text-xl font-bold tracking-tight text-slate-100 sm:text-2xl">
        {title}
      </h1>

      <p className="mt-1.5 text-xs text-slate-400">
        {subtitle}
      </p>
    </div>
  );
}

export function PrimaryButton({
  children,
  loading,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  loading?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      {...props}
      disabled={props.disabled || loading}
      className="flex h-11 w-full items-center justify-center rounded-xl bg-emerald-500 text-xs font-semibold text-slate-950 transition-colors hover:bg-emerald-400 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
    >
      {loading ? "Please wait…" : children}
    </button>
  );
}
