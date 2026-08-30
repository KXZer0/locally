// The Code tab: OpenCode, and coding mode.
//
// In here: launching the real OpenCode TUI with locally added as a
// runtime-only provider, and the coding-mode toggle that keeps the audio and
// utility slots unloaded.
// Not in here: any claim that freeing memory makes a local model able to code.
// The toggle says when it still cannot, because tool calling is GPU/CPU only
// and an agent needs far more context than a small model has.

import { opencodeLaunchBtn, opencodeStatus } from '../core/dom.js';
import { checkHealth } from '../system/bootstrap.js';

// --- OpenCode launcher --------------------------------------------------
// OpenCode is a terminal application. The server opens the real TUI and adds
// locally as a runtime-only provider; the user's normal config remains intact.
export async function launchOpenCode(button, statusEl) {
    button.disabled = true;
    const label = button.textContent;
    button.textContent = 'Opening…';
    statusEl.textContent = 'Opening OpenCode in a terminal…';
    try {
        const workspace = document.getElementById('code-workspace')?.value.trim() || '';
        const resp = await fetch('/v1/opencode/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workspace }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            statusEl.textContent = data.error?.message || 'Could not open OpenCode.';
            return;
        }
        const local = data.local_agent_ready
            ? `locally/${data.local_model} is ready.`
            : `OpenCode is ready; locally is optional (${data.warning}).`;
        statusEl.textContent = `Opened OpenCode ${data.version || ''}. ${local}`.trim();
    } catch (err) {
        statusEl.textContent = err.message;
    } finally {
        button.disabled = false;
        button.textContent = label;
    }
}

opencodeLaunchBtn.addEventListener('click', () => launchOpenCode(opencodeLaunchBtn, opencodeStatus));
// --- Code tab: OpenCode -------------------------------------------------
export const codePanelStatus = document.getElementById('code-panel-status');
export const codeStartBtn = document.getElementById('code-start-btn');

export async function enterCodeTab() {
    codePanelStatus.textContent = 'Checking OpenCode…';
    try {
        const response = await fetch('/v1/opencode/status');
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || 'Could not check OpenCode.');
        if (!data.installed) {
            codePanelStatus.textContent = 'OpenCode is not installed.';
            return;
        }
        const local = data.local_agent_ready
            ? `locally/${data.local_model} is available.`
            : `Your other providers remain available; locally is not agent-ready yet (${data.local_reason}).`;
        codePanelStatus.textContent = `OpenCode ${data.version || 'installed'}. ${local}`;
    } catch (err) {
        codePanelStatus.textContent = err.message;
    }
}

codeStartBtn.addEventListener('click', () => launchOpenCode(codeStartBtn, codePanelStatus));
// --- Coding mode --------------------------------------------------------
// A mode, not a button: the audio and utility slots reload themselves the
// moment anything touches them (entering Voice calls /v1/audio/warm and pulls
// two of them back), so a one-shot free would be undone by the next click with
// nothing to explain why the memory returned.
export const codingModeBtn = document.getElementById('coding-mode-btn');
export const codingModeStatus = document.getElementById('coding-mode-status');

export function paintCodingMode(state, extra) {
    const on = !!state.enabled;
    codingModeBtn.setAttribute('aria-pressed', String(on));
    codingModeBtn.classList.toggle('active', on);
    let line;
    if (!on) {
        line = 'Off — speech, voice and the utility engines load as needed.';
    } else {
        const freed = extra && typeof extra.returned_mb === 'number'
            ? ` Freed ${(extra.returned_mb / 1024).toFixed(1)} GB.` : '';
        line = `On — speech, voice and the utility engines stay unloaded.${freed}`;
    }
    // Freeing memory is not what makes a local model able to code, so say when
    // it still cannot: tool calling is GPU/CPU only and an agent needs ~70k of
    // context. A toggle reading "on" over a model that cannot call a tool
    // would be telling the user something untrue.
    if (state.reason) line += ` ⚠ ${state.reason}`;
    codingModeStatus.textContent = line;
}

codingModeBtn.addEventListener('click', async () => {
    const turningOn = codingModeBtn.getAttribute('aria-pressed') !== 'true';
    codingModeBtn.disabled = true;
    try {
        const resp = await fetch('/v1/coding-mode', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: turningOn }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error?.message || 'could not switch coding mode');
        paintCodingMode(data, data);
        await checkHealth();
    } catch (err) {
        codingModeStatus.textContent = err.message;
    } finally {
        codingModeBtn.disabled = false;
    }
});

fetch('/v1/coding-mode').then(r => r.json()).then(paintCodingMode).catch(() => {});
