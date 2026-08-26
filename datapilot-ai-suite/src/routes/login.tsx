import { createFileRoute } from "@tanstack/react-router";
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout";
import { LoginForm } from "@/components/auth/LoginForm";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in to DataPilot — AI Excel Copilot" },
      {
        name: "description",
        content:
          "Sign in to your DataPilot workspace to analyze spreadsheets, ask questions in natural language, and generate AI insights.",
      },
      { property: "og:title", content: "Sign in to DataPilot" },
      {
        property: "og:description",
        content: "Access your DataPilot workspace and turn spreadsheets into insights.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  return (
    <AuthSplitLayout>
      <LoginForm />
    </AuthSplitLayout>
  );
}
