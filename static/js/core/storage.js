// Persisted settings: the one-time key rename, and nothing else.
//
// In here: work that must happen before the first localStorage read anywhere
// in the app. core/dom.js imports this first for exactly that reason.
// Not in here: reading or writing any individual setting. Each feature owns
// its own keys; a central table of them would be a second place to forget.

// --- Saved settings: carry them across the rename --------------------------
// Preferences used to live under `nollama-*`. Renaming the keys without moving
// the values would silently reset every system prompt, voice choice and
// turn-taking setting on first load after an update, which reads as data loss
// rather than as a rename. Runs once, before any getItem below.
(function migrateStoredSettings() {
    const KEYS = ['system-prompt', 'voice-prompt', 'voice', 'voice-think',
                  'no-think', 'asr-lang', 'code-dir', 'vad', 'vad-gate',
                  'vad-patience'];
    try {
        for (const k of KEYS) {
            const from = `nollama-${k}`, to = `locally-${k}`;
            const old = localStorage.getItem(from);
            // An empty string is a deliberate opt-out and must survive, so the
            // test is against null, not falsiness.
            if (old !== null && localStorage.getItem(to) === null) {
                localStorage.setItem(to, old);
            }
            localStorage.removeItem(from);
        }
    } catch { /* private mode, storage disabled -- defaults are fine */ }
})();
