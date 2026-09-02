"use client";

import { useState, type ChangeEvent, type FormEvent } from "react";
import { Lock, Mail } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { loginWithEmailPassword, loginWithGoogle } from "@/lib/auth/client";

import AuthField from "./AuthField";
import {
  AuthHeading,
  GoogleButton,
  OrDivider,
  PrimaryButton,
} from "./AuthExtras";

type LoginValues = {
  email: string;
  password: string;
};

type LoginErrors = {
  email?: string | undefined;
  password?: string | undefined;
  general?: string | undefined;
};

export function LoginForm() {
  const router = useRouter();
  const [values, setValues] = useState<LoginValues>({
    email: "",
    password: "",
  });

  const [errors, setErrors] = useState<LoginErrors>({});
  const [loading, setLoading] = useState(false);

  const handleChange =
    (field: keyof LoginValues) => (event: ChangeEvent<HTMLInputElement>) => {
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

  const validate = (): boolean => {
    const nextErrors: LoginErrors = {};

    if (!values.email.trim()) {
      nextErrors.email = "Email address is required.";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email)) {
      nextErrors.email = "Enter a valid email address.";
    }

    if (!values.password) {
      nextErrors.password = "Password is required.";
    }

    setErrors(nextErrors);

    return !nextErrors.email && !nextErrors.password;
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!validate()) {
      return;
    }

    setLoading(true);

    try {
      const res = await loginWithEmailPassword(values.email, values.password);

      if (!res.ok) {
        setErrors((prev) => ({
          ...prev,
          general: res.error || "Failed to sign in. Please check your credentials.",
        }));
      } else {
        router.push("/home");
        router.refresh();
      }
    } catch (err: unknown) {
      setErrors((prev) => ({
        ...prev,
        general: err instanceof Error ? err.message : "Failed to sign in. Please try again.",
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
        title="Welcome back"
        subtitle="Sign in to continue to your DataPilot workspace."
      />

      {errors.general && (
        <div className="mb-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-600">
          {errors.general}
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate className="space-y-4">
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
            autoComplete="current-password"
            placeholder="Enter your password"
            value={values.password}
            onChange={handleChange("password")}
            error={errors.password}
          />
        </div>

        <PrimaryButton type="submit" loading={loading}>
          Sign In
        </PrimaryButton>
      </form>

      <OrDivider />

      <GoogleButton onClick={handleGoogle} />

      <p className="mt-7 text-center text-sm text-slate-500">
        Don&apos;t have an account?{" "}
        <Link
          href="/register"
          className="font-semibold text-[var(--auth-accent)] hover:underline"
        >
          Sign up
        </Link>
      </p>
    </div>
  );
}

export default LoginForm;
