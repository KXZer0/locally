// Pure string formatting. No DOM, no fetch, no state.
//
// In here: functions that take a value and return a string, used from more
// than one feature.
// Not in here: anything that touches an element or the network. If a helper
// needs either, it belongs with the feature that owns it.

export const HTML_ESCAPES = Object.freeze({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
});
export function escapeHtml(text) {
    return String(text == null ? '' : text).replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);
}
export function estimateTokensIn(text) {
    return Math.max(1, Math.ceil((text || '').length / 4));
}
export function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
// Model directory names get long ("Qwen2.5-Coder-14B-Instruct"); the rail has
// room for the identity, not the full spec.
export function shortModel(name) {
    if (!name) return '';
    return name.length > 22 ? name.slice(0, 21) + '…' : name;
}
export function hostOf(url) {
    try { return new URL(url).hostname.replace(/^www\./, ''); }
    catch { return url; }
}
