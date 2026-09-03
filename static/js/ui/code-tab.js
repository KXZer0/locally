// The Code tab: OpenCode's web interface, its TUI, and coding mode.
//
// In here: showing the OpenCode WEB interface inside this tab, launching the
// real OpenCode TUI with locally added as a runtime-only provider, and the
// coding-mode toggle that keeps the audio and utility slots unloaded.
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
        // Already serving -- show it rather than asking for a click that would
        // only discover the same thing.
        if (data.web?.running && data.web.url && viewingFromServerMachine()) {
            showWebView(data.web.url);
        } else {
            showLauncher();
            if (!viewingFromServerMachine()) {
                codeWebBtn.disabled = true;
                codeWebBtn.title = 'The OpenCode server is loopback-only; open '
                    + 'locally on its own machine to use the web interface.';
            }
        }
    } catch (err) {
        codePanelStatus.textContent = err.message;
    }
}

codeStartBtn.addEventListener('click', () => launchOpenCode(codeStartBtn, codePanelStatus));

// --- OpenCode web interface ---------------------------------------------
// The web UI is what shows plans, sessions and diffs, so it is the default
// view of this tab. It is embedded rather than linked because OpenCode allows
// it: measured on 1.18.25, no frame-ancestors, no X-Frame-Options, no HSTS.
// Odysseus sends all three, which is why that one is a sidebar link instead.
export const codeWeb = document.getElementById('code-web');
export const codeWebBtn = document.getElementById('code-web-btn');
export const codeEmpty = document.getElementById('code-empty');
export const codeWebFrame = document.getElementById('code-web-frame');
export const codeWebUrl = document.getElementById('code-web-url');
export const codeWebStop = document.getElementById('code-web-stop');
export const codeWebExternal = document.getElementById('code-web-external');

let webUrl = null;

// The OpenCode server is bound to loopback on purpose -- it runs commands and
// starts unauthenticated, so giving it locally's 0.0.0.0 reach would hand a
// shell to every device on the network. The consequence is that its URL only
// resolves for a browser on the server's own machine: from a phone,
// 127.0.0.1 is the phone. Saying so beats showing a frame that can only ever
// be blank.
export function viewingFromServerMachine() {
    return ['localhost', '127.0.0.1', '::1', '[::1]'].includes(location.hostname);
}

function showWebView(url) {
    webUrl = url;
    codeWebUrl.textContent = url;
    // Assign only when it changes: rewriting src on every status poll would
    // reload the app and throw away whatever session the user was reading.
    if (codeWebFrame.getAttribute('src') !== url) codeWebFrame.setAttribute('src', url);
    codeEmpty.hidden = true;
    codeWeb.hidden = false;
}

function showLauncher() {
    webUrl = null;
    codeWeb.hidden = true;
    codeEmpty.hidden = false;
    // Drop the frame so a stopped server is not left as a dead rendered page.
    codeWebFrame.removeAttribute('src');
}

export async function openCodeWeb() {
    if (!viewingFromServerMachine()) {
        codePanelStatus.textContent =
            'The OpenCode server only listens on the loopback address of the '
            + 'machine running locally, so its interface cannot load from this '
            + 'device. Open locally on that machine to use it.';
        return;
    }
    codeWebBtn.disabled = true;
    const label = codeWebBtn.textContent;
    codeWebBtn.textContent = 'Starting…';
    codePanelStatus.textContent = 'Starting the OpenCode server…';
    try {
        const workspace = document.getElementById('code-workspace')?.value.trim() || '';
        const resp = await fetch('/v1/opencode/web', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workspace }),
        });
        const data = await resp.json();
        if (!resp.ok || !data.url) {
            codePanelStatus.textContent = data.error?.message || 'Could not start OpenCode.';
            return;
        }
        showWebView(data.url);
        codePanelStatus.textContent = data.local_agent_ready
            ? `locally/${data.local_model} is ready.`
            : `Running; locally is optional (${data.warning}).`;
    } catch (err) {
        codePanelStatus.textContent = err.message;
    } finally {
        codeWebBtn.disabled = false;
        codeWebBtn.textContent = label;
    }
}

codeWebBtn.addEventListener('click', openCodeWeb);

codeWebStop.addEventListener('click', async () => {
    codeWebStop.disabled = true;
    try {
        const resp = await fetch('/v1/opencode/web', { method: 'DELETE' });
        const data = await resp.json();
        // Three outcomes, not two. "Still running after we tried" is its own
        // case and must not be painted as either success or "not ours" -- the
        // server verifies the port before claiming a stop, so believe it.
        if (data.running) {
            codePanelStatus.textContent = data.stopped
                ? 'The OpenCode server is still answering; it may have restarted.'
                : 'That server was not started by locally, so it was left running.';
            return;
        }
        showLauncher();
        codePanelStatus.textContent = data.stopped
            ? 'Stopped the OpenCode server.'
            : 'The OpenCode server was already stopped.';
    } catch (err) {
        codePanelStatus.textContent = err.message;
    } finally {
        codeWebStop.disabled = false;
    }
});

// The UI may be running inside a PWA window, so a target="_blank" would open
// in that window's browser rather than the user's default. The server owns
// this for the same reason /v1/open exists.
codeWebExternal.addEventListener('click', () => {
    if (!webUrl) return;
    fetch('/v1/open', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: webUrl }),
    }).catch(() => {});
});
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
