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
      <label className="block text-sm font-medium text-[var(--auth-text)]">
        {label}
      </label>

      <div className="relative">
        <Icon
          size={18}
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
          className={`w-full rounded-xl border bg-white py-3 pl-11 pr-11 text-sm text-[var(--auth-input-text)] outline-none transition-all placeholder:text-[var(--auth-input-placeholder)] ${
            error
              ? "border-red-400 focus:border-red-500 focus:ring-2 focus:ring-red-100"
              : "border-[var(--auth-input-border)] focus:border-[var(--auth-accent)] focus:ring-2 focus:ring-blue-100"
          }`}
        />

        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword((prev) => !prev)}
            className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-400 transition-colors hover:text-slate-600"
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        )}
      </div>

      {error && (
        <p className="text-xs font-medium text-red-500">
          {error}
        </p>
      )}
    </div>
  );
}

export default AuthField;
