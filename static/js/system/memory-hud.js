// The bottom-right memory panel.
//
// In here: the machine's RAM picture, the GPU's advertised ceiling shown
// beside what the machine can actually back, per-slot weights and offload, and
// the two buttons that free them. Thresholds come from the server's own
// _usable_gpu_bytes(), so the HUD cannot contradict the offload decision it
// is describing.
// Not in here: the context ring, which shares the same surface but measures
// the prompt, not the machine (chat/context.js).

import { scheduleExactContextCount } from '../chat/context.js';
import { memAction, memBarFill, memFreeBtn, memHeadline, memHud, memHudPill, memMessage, memRows, swapStatus, unloadBtn } from '../core/dom.js';
import { checkHealth } from './bootstrap.js';
import { closeContextModelMenu } from './swap.js';

export const gbStr = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB');

// Two claims, kept apart on purpose. What we dropped is ours and auditable
// (weights on disk). What the machine now has free is an observation about
// the whole box — under memory pressure Windows dumps standby pages at the
// same moment, and we once saw +22.8 GB for 6.6 GB of models. Reporting that
// as "locally freed 22.3 GB" would be a lie, so it's stated as a reading.
export function freedPhrase(mem) {
    const parts = [];
    if (typeof mem?.weights_mb === 'number') {
        parts.push(` (${gbStr(mem.weights_mb)} of weights)`);
    }
    const before = mem?.before?.system_available_mb;
    const after = mem?.after?.system_available_mb;
    if (typeof before === 'number' && typeof after === 'number') {
        parts.push(`. Free memory on this machine: ${gbStr(before)} → ${gbStr(after)}`);
    }
    return parts.join('');
}
unloadBtn.addEventListener('click', async () => {
    unloadBtn.disabled = true;
    const label = unloadBtn.innerHTML;
    unloadBtn.textContent = 'Freeing…';
    try {
        const resp = await fetch('/v1/models/unload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        const data = await resp.json();
        if (resp.ok && !data.unloaded.length) {
            swapStatus.textContent = 'Nothing was loaded — the memory is '
                + 'already free. Your next message loads a model again.';
        } else if (resp.ok) {
            const names = data.unloaded.map(u => u.model).join(', ');
            swapStatus.textContent =
                `Unloaded ${names}${freedPhrase(data.memory)}. `
                + 'Reloads automatically on your next message.';
        } else {
            swapStatus.textContent = data.error?.message || 'Unload failed';
        }
        await checkHealth();
    } catch (err) {
        swapStatus.textContent = err.message;
    } finally {
        unloadBtn.disabled = false;
        unloadBtn.innerHTML = label;
    }
});
// --- Memory HUD -------------------------------------------------------------
// Every number here already existed server-side and none of it was shown, which
// is how a model quietly paging at 0.5 tok/s read as "the model is slow" and how
// an iGPU's advertised ceiling read as dedicated VRAM. Thresholds come from the
// server's own _usable_gpu_bytes(), so the HUD cannot disagree with the offload
// decision it is describing.

export let lastMemoryRenderSignature = '';

export const asGb = mb => (mb == null ? null : mb / 1024);
export const gbText = mb => (mb == null ? '—' : `${(mb / 1024).toFixed(1)} GB`);

export function renderMemory(m) {
    if (!m || !m.system || m.system.total_mb == null) {
        lastMemoryRenderSignature = '';
        memHud.dataset.memory = 'unavailable';
        memHeadline.textContent = 'Memory unavailable';
        memMessage.textContent = 'Context remains available; machine memory statistics could not be read.';
        memAction.hidden = true;
        memRows.innerHTML = '';
        memBarFill.style.transform = 'scaleX(0)';
        return;
    }
    const renderSignature = JSON.stringify(m);
    if (renderSignature === lastMemoryRenderSignature) return;
    lastMemoryRenderSignature = renderSignature;
    memHud.dataset.memory = 'ready';
    memHud.dataset.state = m.state;

    const total = m.system.total_mb;
    const avail = m.system.available_mb ?? 0;
    const used = Math.max(0, total - avail);
    const usedScale = total > 0 ? Math.min(1, used / total) : 0;
    memBarFill.style.transform = `scaleX(${usedScale.toFixed(3)})`;

    // The collapsed line stays boring on purpose: two numbers, plus the offload
    // percentage only when there is one, because that is the figure that
    // explains a slow model.
    const offload = (m.slots || []).find(s => s.offload_ratio);
    memHeadline.textContent =
        `${(used / 1024).toFixed(1)}/${(total / 1024).toFixed(1)} GB`
        + (offload ? ` · offload ${offload.offload_ratio}%` : '');

    memMessage.textContent = m.message || '';
    memAction.textContent = m.action || '';
    memAction.hidden = !m.action;

    const rows = [['Free RAM', gbText(avail)]];
    if (m.gpu && m.gpu.driver_ceiling_mb != null) {
        if (m.gpu.shares_system_ram) {
            // The two rows that exist to kill "I have 24 GB of VRAM". An
            // integrated GPU's ceiling is a policy share of the same RAM
            // everything else is using; showing it alone invites the wrong
            // conclusion, so it never appears without its real counterpart.
            rows.push(['GPU advertises', gbText(m.gpu.driver_ceiling_mb), 'is-claim']);
            rows.push(['Actually usable', gbText(m.gpu.usable_mb), 'is-real']);
        } else {
            rows.push(['GPU memory', gbText(m.gpu.driver_ceiling_mb)]);
        }
    }
    for (const s of m.slots || []) {
        rows.push([
            s.model,
            `${s.device} · ${gbText(s.weights_mb)}`
            + (s.offload_ratio ? ` · ${s.offload_ratio}% from disk` : ''),
        ]);
    }

    memRows.innerHTML = '';
    for (const [label, value, cls] of rows) {
        const dt = document.createElement('dt');
        dt.textContent = label;
        const dd = document.createElement('dd');
        dd.textContent = value;
        if (cls) dd.classList.add(cls);
        memRows.append(dt, dd);
    }
}

export async function refreshMemory() {
    try {
        const resp = await fetch('/v1/memory');
        if (resp.ok) renderMemory(await resp.json());
        else renderMemory(null);
    } catch { renderMemory(null); }
}

memHudPill.addEventListener('click', () => {
    const opening = !memHud.classList.contains('pinned');
    memHud.classList.toggle('pinned', opening);
    memHudPill.setAttribute('aria-expanded', String(opening));
    if (opening) scheduleExactContextCount(0);
    else closeContextModelMenu();
});

memHud.addEventListener('mouseenter', () => scheduleExactContextCount(0));
memHudPill.addEventListener('focus', () => scheduleExactContextCount(0));
document.addEventListener('pointerdown', (event) => {
    if (!memHud.classList.contains('pinned') || memHud.contains(event.target)) return;
    memHud.classList.remove('pinned');
    memHudPill.setAttribute('aria-expanded', 'false');
    closeContextModelMenu();
});
document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !memHud.classList.contains('pinned')) return;
    memHud.classList.remove('pinned');
    memHudPill.setAttribute('aria-expanded', 'false');
    closeContextModelMenu();
    memHudPill.focus();
});

memFreeBtn.addEventListener('click', async () => {
    memFreeBtn.disabled = true;
    const label = memFreeBtn.innerHTML;
    memFreeBtn.textContent = 'Freeing…';
    try {
        const resp = await fetch('/v1/models/unload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        const data = await resp.json();
        memMessage.textContent = resp.ok
            ? (data.unloaded.length
                ? `Unloaded ${data.unloaded.map(u => u.model).join(', ')}`
                  + `${freedPhrase(data.memory)}. Reloads on your next message.`
                : 'Nothing was loaded — the memory is already free.')
            : (data.error?.message || 'Unload failed');
        await checkHealth();
        await refreshMemory();
    } catch (err) {
        memMessage.textContent = err.message;
    } finally {
        memFreeBtn.disabled = false;
        memFreeBtn.innerHTML = label;
    }
});
