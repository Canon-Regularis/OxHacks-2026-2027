export const SETTINGS_KEY = 'gridpulse:settings:v1';
export const ACCENTS = {
  orange: { label: 'Pastel orange', color: '#e6b080' },
  red: { label: 'Pastel red', color: '#e6a0a0' },
  green: { label: 'Pastel green', color: '#a8c9ad' },
} as const;
export type Accent = keyof typeof ACCENTS;
export interface SettingsPreference {
  version: 1;
  accent: Accent;
  timeDisplay: 'utc' | 'local';
  motion: 'system' | 'reduce';
  gridlines: boolean;
  lineThickness: 'standard' | 'bold';
  rememberView: boolean;
}
export function defaultSettings(): SettingsPreference {
  return {
    version: 1,
    accent: 'orange',
    timeDisplay: 'utc',
    motion: 'system',
    gridlines: true,
    lineThickness: 'standard',
    rememberView: true,
  };
}
export function normalizeSettings(value: unknown): SettingsPreference {
  const defaults = defaultSettings();
  if (!value || typeof value !== 'object' || Array.isArray(value))
    return defaults;
  const p = value as Record<string, unknown>;
  if (p.version !== 1) return defaults;
  return {
    ...defaults,
    accent: p.accent === 'red' || p.accent === 'green' ? p.accent : 'orange',
    timeDisplay: p.timeDisplay === 'local' ? 'local' : 'utc',
    motion: p.motion === 'reduce' ? 'reduce' : 'system',
    gridlines:
      typeof p.gridlines === 'boolean' ? p.gridlines : defaults.gridlines,
    lineThickness: p.lineThickness === 'bold' ? 'bold' : 'standard',
    rememberView:
      typeof p.rememberView === 'boolean'
        ? p.rememberView
        : defaults.rememberView,
  };
}
export function readSettings(
  storage: Pick<Storage, 'getItem'>,
): SettingsPreference {
  try {
    return normalizeSettings(
      JSON.parse(storage.getItem(SETTINGS_KEY) ?? 'null'),
    );
  } catch {
    return defaultSettings();
  }
}
function zoneLabel(
  iso: string,
  display: SettingsPreference['timeDisplay'],
): string {
  if (display === 'utc') return 'UTC';
  const offset = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Europe/London',
    timeZoneName: 'shortOffset',
  })
    .formatToParts(new Date(iso))
    .find((part) => part.type === 'timeZoneName')?.value;
  return offset === 'GMT+1' ? 'BST' : 'GMT';
}
export function formatTime(
  iso: string,
  display: SettingsPreference['timeDisplay'],
  includeZone = true,
): string {
  const time = new Date(iso).toLocaleTimeString('en-GB', {
    timeZone: display === 'local' ? 'Europe/London' : 'UTC',
    hour: '2-digit',
    minute: '2-digit',
  });
  return includeZone ? `${time} ${zoneLabel(iso, display)}` : time;
}
export function formatDateTime(
  iso: string,
  display: SettingsPreference['timeDisplay'],
): string {
  const date = new Date(iso).toLocaleString('en-GB', {
    timeZone: display === 'local' ? 'Europe/London' : 'UTC',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  return `${date} ${zoneLabel(iso, display)}`;
}
