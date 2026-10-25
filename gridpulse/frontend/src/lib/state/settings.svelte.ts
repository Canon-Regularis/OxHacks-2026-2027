import { getContext } from 'svelte';
import { defaultSettings, type SettingsPreference } from './settings';
export interface SettingsState extends SettingsPreference {
  systemReduced: boolean;
}
export function createSettingsState(): SettingsState {
  const state = $state({
    ...defaultSettings(),
    systemReduced: true,
  });
  return state;
}
export function useSettings(): SettingsState {
  return getContext<SettingsState>('gridpulse:settings');
}
export function reducedMotion(settings: SettingsState): boolean {
  return settings.motion === 'reduce' || settings.systemReduced;
}
