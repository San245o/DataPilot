import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout";
import { LoginForm } from "@/components/auth/LoginForm";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Sign in to DataPilot — AI Excel Copilot",
  description:
    "Sign in to your DataPilot workspace to analyze spreadsheets, ask questions in natural language, and generate AI insights.",
};

export default function LoginPage() {
  return (
    <AuthSplitLayout>
      <LoginForm />
    </AuthSplitLayout>
  );
}
