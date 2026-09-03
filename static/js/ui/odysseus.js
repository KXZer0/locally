// Odysseus: a link in the sidebar, and the check that decides whether to show it.
//
// In here: the configurable address, its reachability probe, and opening it.
// Not in here: any attempt to embed it. docs/ODYSSEUS.md records why, and it
// is not a matter of taste -- three mechanisms each break an embed on their
// own: HSTS upgrades the frame to https that a local deployment does not
// serve; cookies scoped to its own hostname are third-party inside our origin
// and get partitioned or dropped, logging you out on every navigation; and its
// service worker expects to control a top-level page. Fighting all three buys
// a worse version of a link. This is the opposite call to the OpenCode web
// interface in code-tab.js, which sets none of those headers and so is framed.

import { odysseusNav, odysseusLink, odysseusUrlInput, odysseusStatus, odysseusDot, odysseusHeadline, odysseusAction, odysseusAutostart } from '../core/dom.js';

const KEY = 'locally-odysseus-url';
export const ODYSSEUS_DEFAULT = 'http://localhost:7000';

export function odysseusUrl() {
    // null means "never set" -> use the default. An empty string is a
    // deliberate opt-out and must survive a reload, the same rule the system
    // prompt follows.
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch { /* storage disabled */ }
    if (saved === null) return ODYSSEUS_DEFAULT;
    return saved.trim();
}

function saveOdysseusUrl(value) {
    try { localStorage.setItem(KEY, value); } catch { /* storage disabled */ }
}

// Reachability is tested FROM THE BROWSER, not from the server, because the
// browser is what will follow the link. A server-side probe would light the
// entry up whenever locally can reach Odysseus -- including when the person
// looking at this page is on a phone that cannot, which is exactly the
// "control wired to nothing" this is meant to avoid.
//
// `no-cors` is what makes it possible at all: Odysseus sends no CORS headers
// for us, so a normal fetch is unreadable. An opaque response still resolves
// on success and rejects on a refused connection, and reachable/not is the
// only bit we need.
export async function probeOdysseus(url, timeoutMs = 2500) {
    if (!url) return false;
    let parsed;
    try { parsed = new URL(url); } catch { return false; }
    if (!['http:', 'https:'].includes(parsed.protocol)) return false;
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), timeoutMs);
    try {
        await fetch(parsed.origin, { mode: 'no-cors', signal: abort.signal, cache: 'no-store' });
        return true;
    } catch {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

// What the SERVER knows, which the browser cannot see: whether a checkout
// exists, whether Docker is up, and whether the process is ours. Without this
// the panel could only ever say "not answering", which is the same sentence for
// "you never installed it", "Docker Desktop is closed" and "it is still
// booting" -- three problems with three different fixes.
async function serverState() {
    try {
        const resp = await fetch('/v1/odysseus');
        if (!resp.ok) return null;
        return await resp.json();
    } catch {
        return null;
    }
}

// state -> [dot, headline, action button label or null, hint]
function present(url, reachable, srv) {
    if (reachable || srv?.running) {
        const who = srv?.managed ? ' · started by locally' : '';
        return ['running', `Running on ${url}${who}`, srv?.managed ? 'Stop' : null,
                'Open it from the sidebar entry.'];
    }
    if (!srv) {
        return ['absent', 'Not answering',
                null, `Nothing is answering at ${url}. Start Odysseus, or correct the address.`];
    }
    if (!srv.installed) {
        return ['absent', 'Not installed', null,
                'Clone Odysseus next to this folder (or set ODYSSEUS_DIR) and it '
                + 'will be found automatically. See docs/ODYSSEUS.md.'];
    }
    if (!srv.docker_available) {
        // The engine is NAMED from the server rather than assumed to be Docker.
        // A sleeping Podman VM is a healthy idle state and stays startable;
        // Docker still needs its separately-owned daemon to be brought up.
        const engine = srv.engine || null;
        if (srv.docker_installed) {
            const name = engine === 'podman' ? 'Podman' : 'Docker';
            if (engine === 'podman') {
                return ['stopped', 'Installed, sleeping', 'Start',
                        'Podman will start quietly on demand. Its VM stops again '
                        + 'when locally stops Odysseus and no other containers are running.'];
            }
            const fix = 'Start Docker Desktop, then start it here.';
            return ['error', `${name} is not running`, null,
                    `Odysseus runs in ${name}. ${fix}`];
        }
        return ['absent', 'No container engine', null,
                'Odysseus runs as a Compose stack. Install Podman or Docker '
                + 'Desktop, then this can start it for you.'];
    }
    return ['stopped', 'Installed, not running', 'Start',
            'Bring it up now, or turn on "Start it with locally" below.'];
}

let pending = false;

export async function refreshOdysseus() {
    const url = odysseusUrl();
    if (odysseusUrlInput && document.activeElement !== odysseusUrlInput) {
        odysseusUrlInput.value = url;
    }
    if (odysseusHeadline && !pending) odysseusHeadline.textContent = 'Checking…';

    const [reachable, srv] = await Promise.all([
        url ? probeOdysseus(url) : Promise.resolve(false),
        serverState(),
    ]);

    const [dot, headline, action, hint] = present(url, reachable, srv);
    if (odysseusDot) odysseusDot.dataset.state = dot;
    if (odysseusHeadline) odysseusHeadline.textContent = headline;
    if (odysseusStatus) odysseusStatus.textContent = hint;
    if (odysseusAction) {
        odysseusAction.hidden = !action;
        if (action) odysseusAction.textContent = action;
    }
    if (odysseusAutostart && srv && typeof srv.autostart === 'boolean') {
        odysseusAutostart.checked = srv.autostart;
    }
    // Hidden rather than disabled: unlike a utility engine, whose greyed-out
    // button teaches that the task exists on the other engine, a dead link to
    // software the user may simply not run has nothing to teach. Settings
    // always says what happened, so the state is never unexplained.
    if (odysseusNav) odysseusNav.hidden = !(reachable || srv?.running);
    return reachable;
}

// Start and stop are the same button because they are the same question asked
// in two states, and two buttons where one is always dead is the control-wired-
// to-nothing problem again.
async function runAction() {
    const stopping = odysseusAction.textContent === 'Stop';
    pending = true;
    odysseusAction.disabled = true;
    const was = odysseusAction.textContent;
    odysseusAction.textContent = stopping ? 'Stopping…' : 'Starting…';
    if (odysseusHeadline) {
        odysseusHeadline.textContent = stopping
            ? 'Stopping Odysseus…'
            : 'Starting Odysseus… first run pulls images and can take minutes.';
    }
    try {
        const resp = await fetch(`/v1/odysseus/${stopping ? 'stop' : 'start'}`, { method: 'POST' });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok && odysseusStatus) {
            odysseusStatus.textContent = data.error?.message || data.detail || 'That did not work.';
        }
    } catch (err) {
        if (odysseusStatus) odysseusStatus.textContent = err.message;
    } finally {
        pending = false;
        odysseusAction.disabled = false;
        odysseusAction.textContent = was;
        refreshOdysseus();
    }
}

if (odysseusAction) odysseusAction.addEventListener('click', runAction);

if (odysseusAutostart) {
    odysseusAutostart.addEventListener('change', async () => {
        try {
            await fetch('/v1/odysseus/autostart', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: odysseusAutostart.checked }),
            });
        } catch { /* the refresh below re-reads the truth from the server */ }
        refreshOdysseus();
    });
}

// The UI may be running inside a PWA window, where target="_blank" opens in
// that window's browser rather than the user's default. The server owns this
// for the same reason /v1/open exists at all.
export function openOdysseus() {
    const url = odysseusUrl();
    if (!url) return;
    fetch('/v1/open', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    }).catch(() => window.open(url, '_blank', 'noopener'));
}

if (odysseusLink) odysseusLink.addEventListener('click', openOdysseus);

if (odysseusUrlInput) {
    odysseusUrlInput.addEventListener('change', () => {
        saveOdysseusUrl(odysseusUrlInput.value.trim());
        refreshOdysseus();
    });
}

refreshOdysseus();
