// The right-hand list of pages an answer was built from.
//
// In here: rendering, opening and closing the panel, including the honest
// labels for a page that was only ever a snippet or was trimmed out of the
// synthesis.
// Not in here: the search request (chat/websearch.js). The panel renders what
// it is handed and asks nothing.

import { hostOf } from '../core/format.js';

// --- Sources panel -----------------------------------------------------------

// A right-hand list of real pages, opened when an answer cites the web. The
// inline "[1] [2] [3]" run-on under the answer was footnote-shaped and nobody
// clicks footnotes; a panel with favicons and hostnames reads as "here is what
// this was built from".
//
// Favicons load straight from each site. That is an external request, but the
// page it belongs to was just fetched from that same host to build the answer,
// so it adds no origin that was not already contacted. Offline, or on a host
// with no icon, it falls back to a monogram instead of a broken image.

export const sourcesPanel = document.getElementById('sources-panel');
export const sourcesList = document.getElementById('sources-list');
export const sourcesCount = document.getElementById('sources-count');
export let sourcesCloseTimer = null;

export function renderSources(sources, dropped = []) {
    if (!sourcesPanel || !sourcesList) return;
    sourcesList.textContent = '';
    const droppedBySource = new Map((dropped || []).map(item => [item.source, item]));
    for (const [i, src] of (sources || []).entries()) {
        const host = hostOf(src.url);
        const li = document.createElement('li');
        li.className = 'source-item';

        const a = document.createElement('a');
        a.href = src.url;
        a.target = '_blank';
        a.rel = 'noreferrer noopener';
        // The UI runs inside an Edge PWA window, so target="_blank" opens Edge
        // whatever the user's default browser is. The page cannot escape its
        // own host, so the server does it. Falls through to the normal link if
        // the endpoint is unreachable, rather than dead-ending the click.
        a.addEventListener('click', (ev) => {
            ev.preventDefault();
            fetch('/v1/open', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: src.url }),
            }).then(r => { if (!r.ok) window.open(src.url, '_blank', 'noopener'); })
              .catch(() => window.open(src.url, '_blank', 'noopener'));
        });

        const img = document.createElement('img');
        img.className = 'source-favicon';
        img.alt = '';
        img.loading = 'lazy';
        img.src = `https://${host}/favicon.ico`;
        img.addEventListener('error', () => {
            const mono = document.createElement('span');
            mono.className = 'source-mono';
            mono.textContent = host.charAt(0);
            img.replaceWith(mono);
        });
        a.appendChild(img);

        const copy = document.createElement('span');
        copy.className = 'source-copy';
        const title = document.createElement('span');
        title.className = 'source-title';
        title.textContent = src.title || host;
        const hostEl = document.createElement('span');
        hostEl.className = 'source-host';
        // "snippet only" is the honest label for a page the fetcher could not
        // read. It still informed the answer, but through two lines of search
        // result rather than its actual content.
        const omitted = droppedBySource.get(i + 1);
        hostEl.textContent = src.read === false
            ? `${host} · snippet only`
            : omitted
                ? `${host} · ${omitted.trimmed ? 'passage trimmed' : 'not in synthesis'}`
                : host;
        if (src.read === false) li.classList.add('snippet-only');
        if (omitted) li.classList.add('truncated');
        copy.append(title, hostEl);
        a.appendChild(copy);

        const idx = document.createElement('span');
        idx.className = 'source-index';
        idx.textContent = String(i + 1);
        a.appendChild(idx);

        li.appendChild(a);
        sourcesList.appendChild(li);
    }
    if (sourcesCount) {
        const all = (sources || []).length;
        const read = (sources || []).filter(x => x.read !== false).length;
        sourcesCount.textContent = !all ? ''
            : (read === all ? `${all} page${all === 1 ? '' : 's'}`
                            : `${read} of ${all} read`);
    }
}

export function openSources(sources, dropped) {
    if (!sourcesPanel) return;
    if (sources) renderSources(sources, dropped);
    clearTimeout(sourcesCloseTimer);
    sourcesPanel.hidden = false;
    void sourcesPanel.offsetHeight;          // let the transform start from off-screen
    sourcesPanel.classList.add('open');
    document.body.classList.add('sources-open');
}

export function closeSources() {
    if (!sourcesPanel) return;
    sourcesPanel.classList.remove('open');
    document.body.classList.remove('sources-open');
    const ms = parseFloat(getComputedStyle(sourcesPanel).transitionDuration) * 1000 || 0;
    sourcesCloseTimer = setTimeout(() => { sourcesPanel.hidden = true; }, ms);
}

document.getElementById('sources-close')?.addEventListener('click', closeSources);
