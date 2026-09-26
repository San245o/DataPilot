"use client";

import { useState } from "react";
import type { ChangeEvent, ComponentType } from "react";
import { Eye, EyeOff } from "lucide-react";

type AuthFieldProps = {
  label: string;
  icon: ComponentType<{ size?: number; className?: string }>;
  type?: string;
  autoComplete?: string;
  placeholder?: string;
  value: string;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  error?: string | undefined;
  isPassword?: boolean;
  maxLength?: number | undefined;
};

export function AuthField({
  label,
  icon: Icon,
  type = "text",
  autoComplete,
  placeholder,
  value,
  onChange,
  error,
  isPassword = false,
  maxLength,
}: AuthFieldProps) {
  const [showPassword, setShowPassword] = useState(false);

  const inputType = isPassword
    ? showPassword
      ? "text"
      : "password"
    : type;

  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-medium text-slate-300">
        {label}
      </label>

      <div className="relative">
        <Icon
          size={16}
          className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400"
        />

        <input
          type={inputType}
          autoComplete={autoComplete}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          maxLength={maxLength}
          aria-invalid={!!error}
          className={`w-full rounded-xl border bg-slate-900/80 py-2.5 pl-10 pr-10 text-xs text-slate-100 outline-none transition-all placeholder:text-slate-500 ${
            error
              ? "border-red-800/80 focus:border-red-500 focus:ring-1 focus:ring-red-500/20"
              : "border-slate-800 focus:border-emerald-500/80 focus:ring-1 focus:ring-emerald-500/20"
          }`}
        />

        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword((prev) => !prev)}
            className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-500 transition-colors hover:text-slate-300"
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        )}
      </div>

      {error && (
        <p className="text-[11px] font-medium text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}

export default AuthField;
