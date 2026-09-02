"use client";

import { useState, type ChangeEvent, type FormEvent } from "react";
import { Lock, Mail, ShieldCheck, User } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { signUpWithEmailPassword, loginWithGoogle } from "@/lib/auth/client";

import AuthField from "./AuthField";
import { PasswordStrength, scorePassword } from "./PasswordStrength";

import {
  AuthHeading,
  GoogleButton,
  OrDivider,
  PrimaryButton,
} from "./AuthExtras";

type SignUpValues = {
  fullName: string;
  email: string;
  password: string;
  confirmPassword: string;
};

type SignUpErrors = {
  fullName?: string | undefined;
  email?: string | undefined;
  password?: string | undefined;
  confirmPassword?: string | undefined;
  terms?: string | undefined;
  general?: string | undefined;
};

export function SignUpForm() {
  const router = useRouter();
  const [values, setValues] = useState<SignUpValues>({
    fullName: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [terms, setTerms] = useState(false);
  const [errors, setErrors] = useState<SignUpErrors>({});
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleChange =
    (field: keyof SignUpValues) => (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.target.value;

      setValues((previous) => ({
        ...previous,
        [field]: value,
      }));

      setErrors((previous) => ({
        ...previous,
        [field]: undefined,
        general: undefined,
      }));
    };

  const handleTermsChange = (event: ChangeEvent<HTMLInputElement>) => {
    const checked = event.target.checked;

    setTerms(checked);

    if (checked) {
      setErrors((previous) => ({
        ...previous,
        terms: undefined,
      }));
    }
  };

  const validate = (): boolean => {
    const nextErrors: SignUpErrors = {};

    if (!values.fullName.trim()) {
      nextErrors.fullName = "Full name is required.";
    } else if (values.fullName.trim().length > 100) {
      nextErrors.fullName = "Name must be under 100 characters.";
    }

    if (!values.email.trim()) {
      nextErrors.email = "Email address is required.";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email)) {
      nextErrors.email = "Enter a valid email address.";
    }

    if (!values.password) {
      nextErrors.password = "Password is required.";
    } else if (values.password.length < 8) {
      nextErrors.password = "Use at least 8 characters.";
    } else if (scorePassword(values.password) < 2) {
      nextErrors.password = "Add letters, numbers or symbols.";
    }

    if (!values.confirmPassword) {
      nextErrors.confirmPassword = "Please confirm your password.";
    } else if (values.confirmPassword !== values.password) {
      nextErrors.confirmPassword = "Passwords do not match.";
    }

    if (!terms) {
      nextErrors.terms = "Please accept the Terms of Service and Privacy Policy.";
    }

    setErrors(nextErrors);

    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!validate()) {
      return;
    }

    setLoading(true);

    try {
      const res = await signUpWithEmailPassword(values.fullName, values.email, values.password);

      if (!res.ok) {
        setErrors((prev) => ({
          ...prev,
          general: res.error || "Failed to create account.",
        }));
      } else if (res.message) {
        setSuccessMessage(res.message);
      } else {
        router.push("/home");
        router.refresh();
      }
    } catch (err: unknown) {
      setErrors((prev) => ({
        ...prev,
        general: err instanceof Error ? err.message : "Failed to create account. Please try again.",
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleGoogle = async () => {
    try {
      await loginWithGoogle();
    } catch (err: unknown) {
      setErrors((prev) => ({
        ...prev,
        general: err instanceof Error ? err.message : "Google authentication failed.",
      }));
    }
  };

  return (
    <div>
      <AuthHeading
        title="Create your DataPilot account"
        subtitle="Start turning your spreadsheets into insights."
      />

      {errors.general && (
        <div className="mb-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-600">
          {errors.general}
        </div>
      )}

      {successMessage && (
        <div className="mb-4 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-700">
          {successMessage}
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate className="space-y-4">
        <AuthField
          label="Full Name"
          icon={User}
          autoComplete="name"
          placeholder="Jane Cooper"
          maxLength={100}
          value={values.fullName}
          onChange={handleChange("fullName")}
          error={errors.fullName}
        />

        <AuthField
          label="Email Address"
          icon={Mail}
          type="email"
          autoComplete="email"
          placeholder="you@company.com"
          value={values.email}
          onChange={handleChange("email")}
          error={errors.email}
        />

        <div>
          <AuthField
            label="Password"
            icon={Lock}
            isPassword
            autoComplete="new-password"
            placeholder="Create a password"
            value={values.password}
            onChange={handleChange("password")}
            error={errors.password}
          />

          <PasswordStrength value={values.password} />
        </div>

        <AuthField
          label="Confirm Password"
          icon={ShieldCheck}
          isPassword
          autoComplete="new-password"
          placeholder="Re-enter your password"
          value={values.confirmPassword}
          onChange={handleChange("confirmPassword")}
          error={errors.confirmPassword}
        />

        <div className="pt-1">
          <label className="flex items-start gap-3 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={terms}
              onChange={handleTermsChange}
              className="mt-0.5 h-4 w-4 shrink-0 rounded border-slate-300 accent-[var(--auth-accent)]"
            />

            <span>
              I agree to the Terms of Service and Privacy Policy.
            </span>
          </label>

          {errors.terms && (
            <p className="mt-1.5 text-xs font-medium text-red-500">
              {errors.terms}
            </p>
          )}
        </div>

        <PrimaryButton type="submit" loading={loading}>
          Create Account
        </PrimaryButton>
      </form>

      <OrDivider />

      <GoogleButton onClick={handleGoogle} />

      <p className="mt-7 text-center text-sm text-slate-500">
        Already have an account?{" "}
        <Link
          href="/login"
          className="font-semibold text-[var(--auth-accent)] hover:underline"
        >
          Sign in
        </Link>
      </p>
    </div>
  );
}

export default SignUpForm;
