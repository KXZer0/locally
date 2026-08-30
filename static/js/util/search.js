// The semantic file-search workspace.
//
// In here: building a temporary in-memory index and querying it. The index is
// bounded and process-local by design; nothing here writes to disk.
// Not in here: the ranking. Retrieval is wide and a cross-encoder reorders it
// server-side, which is why a result's rerank_score can sort above a higher
// embedding score.

import { searchDropzone, searchFileSummary, searchFilesInput, searchIndexBtn, searchQuery, searchQueryWrap, searchResults, searchRun, searchStatus } from '../core/dom.js';
import { escapeHtml } from '../core/format.js';
import { syncUtilityTaskAvailability, taskIsAvailable, utilEngine } from './engine.js';

export let searchIndexId = '';
export async function buildSearchIndex() {
    if (!searchFilesInput.files.length || !taskIsAvailable('search')) return;
    searchIndexBtn.disabled = true;
    searchStatus.textContent = `indexing on ${utilEngine.toUpperCase()}`;
    const form = new FormData();
    for (const file of searchFilesInput.files) form.append('files', file, file.name);
    form.append('engine', utilEngine);
    try {
        const response = await fetch('/v1/util/index', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || 'Indexing failed');
        searchIndexId = data.index_id;
        searchStatus.textContent = `${data.chunks} chunks · memory only`;
        searchQueryWrap.hidden = false;
        searchQuery.focus();
    } catch (error) {
        searchStatus.textContent = error.message;
        searchStatus.classList.add('err');
    } finally {
        syncUtilityTaskAvailability();
    }
}

export async function runFileSearch() {
    const query = searchQuery.value.trim();
    if (!searchIndexId || !query) return;
    searchRun.disabled = true;
    searchResults.innerHTML = '';
    searchStatus.textContent = 'searching';
    try {
        const response = await fetch('/v1/util/search', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index_id: searchIndexId, query, top_k: 5 }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || 'Search failed');
        searchResults.innerHTML = data.results.map(item =>
            `<article class="search-hit"><header><span>${escapeHtml(item.source)}</span>`
            + `<span>${(item.score * 100).toFixed(1)}%</span></header>`
            + `<p>${escapeHtml(item.text)}</p></article>`
        ).join('');
        searchStatus.textContent = `${data.results.length} results`;
    } catch (error) {
        searchStatus.textContent = error.message;
        searchStatus.classList.add('err');
    } finally {
        searchRun.disabled = false;
    }
}
searchDropzone.addEventListener('click', () => searchFilesInput.click());
searchFilesInput.addEventListener('change', () => {
    const count = searchFilesInput.files.length;
    searchFileSummary.textContent = count
        ? `${count} file${count === 1 ? '' : 's'} selected`
        : '';
    searchIndexId = '';
    searchQueryWrap.hidden = true;
    searchResults.innerHTML = '';
    syncUtilityTaskAvailability();
});
searchIndexBtn.addEventListener('click', buildSearchIndex);
searchRun.addEventListener('click', runFileSearch);
searchQuery.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); runFileSearch(); }
});
