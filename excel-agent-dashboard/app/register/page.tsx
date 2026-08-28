import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout";
import { SignUpForm } from "@/components/auth/SignUpForm";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Create your DataPilot account",
  description:
    "Create a DataPilot account and start turning complex spreadsheets into actionable, AI-powered insights.",
};

export default function RegisterPage() {
  return (
    <AuthSplitLayout>
      <SignUpForm />
    </AuthSplitLayout>
  );
}
