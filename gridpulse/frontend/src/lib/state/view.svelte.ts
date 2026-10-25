import {
  DEFAULT_SELECTION,
  normalizeSelection,
} from '../../features/overview/model';
export interface ViewPreference {
  version: 1;
  date: string | null;
  location: string | null;
  metrics: string[];
  scanMetric: string;
}
const KEY = 'gridpulse:view:v1';
export function initialViewPreference(): ViewPreference {
  return {
    version: 1,
    date: null,
    location: null,
    metrics: [...DEFAULT_SELECTION],
    scanMetric: 'price',
  };
}
export function readPreference(
  storage: Pick<Storage, 'getItem'>,
): ViewPreference {
  try {
    const p = JSON.parse(storage.getItem(KEY) ?? 'null');
    if (
      p?.version === 1 &&
      (p.date === null || /^\d{4}-\d{2}-\d{2}$/.test(p.date))
    ) {
      const metrics = normalizeSelection(p.metrics);
      return {
        version: 1,
        date: p.date,
        location: typeof p.location === 'string' ? p.location : null,
        metrics,
        scanMetric: metrics.includes(p.scanMetric) ? p.scanMetric : metrics[0],
      };
    }
  } catch {
    /* Browsers can disable storage or retain an older format. */
  }
  return initialViewPreference();
}
export function writePreference(
  storage: Pick<Storage, 'setItem'>,
  preference: ViewPreference,
) {
  try {
    storage.setItem(KEY, JSON.stringify(preference));
  } catch {
    /* Keep the current session usable without storage. */
  }
}
export function createViewState() {
  const state = $state<ViewPreference>(initialViewPreference());
  return state;
}
