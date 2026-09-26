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

export function OrDivider() {
  return (
    <div className="my-5 flex items-center gap-3">
      <span className="h-px flex-1 bg-slate-800" />

      <span className="text-[10px] font-semibold tracking-wider text-slate-500 uppercase">
        or continue with
      </span>

      <span className="h-px flex-1 bg-slate-800" />
    </div>
  );
}

export function GoogleButton({
  onClick,
}: {
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex h-11 w-full items-center justify-center gap-2.5 rounded-xl border border-slate-800 bg-slate-900/60 text-xs font-medium text-slate-200 transition-colors hover:border-slate-700 hover:bg-slate-900 hover:text-white active:scale-[0.99]"
    >
      <svg
        viewBox="0 0 24 24"
        className="h-4 w-4"
        aria-hidden="true"
      >
        <path
          fill="#4285F4"
          d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.5a5.6 5.6 0 0 1-2.4 3.7v3h3.9c2.3-2.1 3.5-5.2 3.5-8.9z"
        />
        <path
          fill="#34A853"
          d="M12 24c3.2 0 5.9-1.1 7.9-2.9l-3.9-3c-1.1.7-2.4 1.2-4 1.2-3.1 0-5.7-2.1-6.6-4.9H1.4v3.1A12 12 0 0 0 12 24z"
        />
        <path
          fill="#FBBC05"
          d="M5.4 14.4a7.2 7.2 0 0 1 0-4.6V6.7H1.4a12 12 0 0 0 0 10.8l4-3.1z"
        />
        <path
          fill="#EA4335"
          d="M12 4.8c1.8 0 3.3.6 4.6 1.8l3.4-3.4C17.9 1.2 15.2 0 12 0A12 12 0 0 0 1.4 6.7l4 3.1c.9-2.9 3.5-5 6.6-5z"
        />
      </svg>

      <span>Google</span>
    </button>
  );
}
