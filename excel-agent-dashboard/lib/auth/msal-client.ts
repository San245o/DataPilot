import { ConfidentialClientApplication, Configuration, CryptoProvider } from "@azure/msal-node";
import { ENTRA_CONFIG, getEntraAuthority, isDevAuthBypass } from "./config";

let _msalInstance: ConfidentialClientApplication | null = null;

export function getMsalApplication(): ConfidentialClientApplication {
  if (_msalInstance) {
    return _msalInstance;
  }

  if (
    (!ENTRA_CONFIG.webClientId || ENTRA_CONFIG.webClientId.includes("placeholder")) &&
    !isDevAuthBypass()
  ) {
    throw new Error(
      "Microsoft Entra External ID configuration (ENTRA_WEB_CLIENT_ID, etc.) is missing. Please configure your .env.local file or enable DEV_AUTH_BYPASS=true for local development."
    );
  }

  const authority = getEntraAuthority();
  const msalConfig: Configuration = {
    auth: {
      clientId: ENTRA_CONFIG.webClientId,
      authority: authority,
      clientSecret: ENTRA_CONFIG.webClientSecret || undefined,
      knownAuthorities: [ENTRA_CONFIG.tenantSubdomain],
    },
    system: {
      loggerOptions: {
        loggerCallback() {},
        piiLoggingEnabled: false,
        logLevel: 0,
      },
    },
  };

  _msalInstance = new ConfidentialClientApplication(msalConfig);
  return _msalInstance;
}

export const cryptoProvider = new CryptoProvider();
