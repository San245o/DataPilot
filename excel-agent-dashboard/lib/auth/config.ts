export const ENTRA_CONFIG = {
  webClientId: process.env.ENTRA_WEB_CLIENT_ID || process.env.NEXT_PUBLIC_ENTRA_WEB_CLIENT_ID || "placeholder-web-client-id",
  webClientSecret: process.env.ENTRA_WEB_CLIENT_SECRET || "",
  tenantId: process.env.ENTRA_TENANT_ID || process.env.NEXT_PUBLIC_ENTRA_TENANT_ID || "placeholder-tenant-id",
  tenantSubdomain: process.env.ENTRA_TENANT_SUBDOMAIN || process.env.NEXT_PUBLIC_ENTRA_TENANT_SUBDOMAIN || "datapilot.ciamlogin.com",
  apiScope: process.env.ENTRA_API_SCOPE || "api://placeholder-api-client-id/access_as_user",
  authSecret: process.env.AUTH_SECRET || "datapilot-entra-external-id-secret-key-32-chars-min",
  userFlow: process.env.ENTRA_USER_FLOW || "SignUpSignIn",
};

export function isDevAuthBypass(): boolean {
  return (
    process.env.NODE_ENV !== "production" &&
    process.env.DEV_AUTH_BYPASS === "true"
  );
}

export function getEntraAuthority(): string {
  const subdomain = ENTRA_CONFIG.tenantSubdomain.includes(".")
    ? ENTRA_CONFIG.tenantSubdomain
    : `${ENTRA_CONFIG.tenantSubdomain}.ciamlogin.com`;

  return `https://${subdomain}/${ENTRA_CONFIG.tenantId}/v2.0`;
}

export function getEntraJwksUrl(): string {
  const subdomain = ENTRA_CONFIG.tenantSubdomain.includes(".")
    ? ENTRA_CONFIG.tenantSubdomain
    : `${ENTRA_CONFIG.tenantSubdomain}.ciamlogin.com`;

  return `https://${subdomain}/${ENTRA_CONFIG.tenantId}/discovery/v2.0/keys`;
}
