import { useState, type ChangeEvent, type FormEvent } from "react";
import { Lock, Mail } from "lucide-react";

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
};

export function LoginForm() {
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
      // TODO: Connect your Login API here.
      // Example:
      //
      // const response = await fetch("/api/auth/login", {
      //   method: "POST",
      //   headers: {
      //     "Content-Type": "application/json",
      //   },
      //   body: JSON.stringify(values),
      // });

      await new Promise((resolve) => setTimeout(resolve, 700));
    } finally {
      setLoading(false);
    }
  };

  const handleGoogle = () => {
    // TODO: Integrate Google authentication here.
    console.log("Google sign-in clicked");
  };

  const handleForgotPassword = () => {
    // TODO: Connect this to your forgot-password page/flow.
    console.log("Forgot password clicked");
  };

  return (
    <div>
      <AuthHeading
        title="Welcome back"
        subtitle="Sign in to continue to your DataPilot workspace."
      />

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

          <div className="mt-2 flex justify-end">
            <button
              type="button"
              onClick={handleForgotPassword}
              className="text-sm font-medium text-[var(--auth-accent)] hover:underline"
            >
              Forgot password?
            </button>
          </div>
        </div>

        <PrimaryButton type="submit" loading={loading}>
          Sign In
        </PrimaryButton>
      </form>

      <OrDivider />

      <GoogleButton onClick={handleGoogle} />

      <p className="mt-7 text-center text-sm text-slate-500">
        Don&apos;t have an account?{" "}
        <a
          href="/signup"
          className="font-semibold text-[var(--auth-accent)] hover:underline"
        >
          Sign up
        </a>
      </p>
    </div>
  );
}

export default LoginForm;