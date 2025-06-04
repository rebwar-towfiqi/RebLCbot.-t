import { NextIntlClientProvider } from 'next-intl';
import type { ReactNode } from 'react';

export function I18nProvider({ children, messages }: { children: ReactNode; messages: Record<string, string> }) {
  return <NextIntlClientProvider messages={messages}>{children}</NextIntlClientProvider>;
}
