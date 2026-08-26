import { createFileRoute } from "@tanstack/react-router";
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout";
import { SignUpForm } from "@/components/auth/SignUpForm";

export const Route = createFileRoute("/signup")({
  head: () => ({
    meta: [
      { title: "Create your DataPilot account" },
      {
        name: "description",
        content:
          "Create a DataPilot account and start turning complex spreadsheets into actionable, AI-powered insights.",
      },
      { property: "og:title", content: "Create your DataPilot account" },
      {
        property: "og:description",
        content: "Start turning your spreadsheets into insights with DataPilot.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: SignUpPage,
});

function SignUpPage() {
  return (
    <AuthSplitLayout>
      <SignUpForm />
    </AuthSplitLayout>
  );
}
