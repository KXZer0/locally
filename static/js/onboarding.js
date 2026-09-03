// First-run setup. The server owns the curated catalog and destinations; this
// file only presents choices and sends catalog IDs back. No repository, URL,
// filename, command, or filesystem path is accepted from the browser.
(() => {
    const shell = document.getElementById('setup-shell');
    const dialog = document.getElementById('setup-dialog');
    const stage = document.getElementById('setup-stage');
    const stepStatus = document.getElementById('setup-step-status');
    const kicker = document.getElementById('setup-kicker');
    const title = document.getElementById('setup-title');
    const lede = document.getElementById('setup-lede');
    const dots = document.getElementById('setup-dots');
    const back = document.getElementById('setup-back');
    const next = document.getElementById('setup-next');
    const close = document.getElementById('setup-close');
    const status = document.getElementById('setup-status');
    const openButton = document.getElementById('setup-open-btn');
    if (!shell || !stage || !openButton) return;

    const steps = [
        { kicker: 'WELCOME', title: 'Made for this machine',
          lede: 'locally measured the hardware available to OpenVINO. Nothing was uploaded.' },
        { kicker: 'ASSISTANT', title: 'Choose its mind',
          lede: 'Curated models first: the best useful fit, not an endless model directory.' },
        { kicker: 'VOICE', title: 'Give it ears and a voice',
          lede: 'Speech recognition, natural replies, turn-taking, and optional speaker identity.' },
        { kicker: 'WEB', title: 'Choose how it reaches the web',
          lede: 'Reading a URL is built in. Private search remains an explicit local install.' },
        { kicker: 'REVIEW', title: 'Your local stack',
          lede: 'Review the choices. Existing files are reused; only missing pieces download.' },
    ];

    let setup = null;
    let catalog = new Map();
    let step = 0;
    let busy = false;
    let finished = false;
    let assistant = null;
    let backend = 'openvino';
    let ollamaModel = null;
    let stt = 'whisper-small';
    const voice = new Set(['kokoro', 'silero-vad', 'wespeaker']);
    let searxng = false;
    let checkUpdates = false;

    const el = (tag, className, text) => {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    };

    const gb = value => {
        const number = Number(value || 0);
        if (!number) return 'size unknown';
        return number < 0.1 ? `${Math.round(number * 1024)} MB` : `${number.toFixed(number < 1 ? 1 : 0)} GB`;
    };

    function selectedIds() {
        const ids = [];
        if (backend === 'openvino' && assistant) ids.push(assistant);
        if (stt) ids.push(stt);
        ids.push(...voice);
        if (searxng) ids.push('searxng');
        return [...new Set(ids)];
    }

    function initializeSelections() {
        const deviceKinds = new Set((setup.devices || []).map(device => device.kind));
        const configured = setup.configured?.assistant || {};
        if (configured.backend === 'ollama' && setup.ollama?.available) {
            backend = 'ollama';
            ollamaModel = configured.model || setup.ollama.models?.[0]?.name || null;
        } else if (!deviceKinds.has('NPU') && !deviceKinds.has('GPU') && setup.ollama?.models?.length) {
            backend = 'ollama';
            ollamaModel = setup.ollama.models[0].name;
        } else {
            backend = 'openvino';
            assistant = setup.recommended?.assistant || 'smollm3-3b';
            const savedPath = configured.model_dir;
            if (savedPath) {
                const match = [...catalog.values()].find(item => item.path === savedPath);
                if (match) assistant = match.id;
            }
        }

        const configuredVoice = setup.configured?.voice || {};
        const savedStt = [...catalog.values()].find(item =>
            item.component === 'stt' && item.path === configuredVoice.stt_dir);
        if (savedStt) stt = savedStt.id;
        for (const component of ['tts', 'vad', 'speaker']) {
            const saved = [...catalog.values()].find(item =>
                item.component === component && item.path === configuredVoice[`${component}_dir`]);
            if (saved) voice.add(saved.id);
        }
        searxng = Boolean(setup.configured?.web?.searxng);
        checkUpdates = Boolean(setup.configured?.web?.check_updates);
    }

    function deviceCard(name, detail, meta) {
        const card = el('article', 'setup-device-card');
        card.append(el('strong', '', name), el('span', '', detail));
        if (meta) card.append(el('span', 'mono', meta));
        return card;
    }

    function renderHardware() {
        const grid = el('div', 'setup-hardware-grid');
        for (const device of setup.devices || []) {
            const note = device.kind === 'NPU' ? 'Low-power assistant engine'
                : device.kind === 'GPU' ? 'Vision and larger models'
                : 'Universal fallback';
            grid.append(deviceCard(device.kind, device.name, note));
        }
        for (const gpu of setup.nvidia || []) {
            grid.append(deviceCard('NVIDIA GPU', gpu.name, `${(gpu.memory_mb / 1024).toFixed(1)} GB · use through Ollama`));
        }
        const total = setup.memory?.system?.total_mb;
        const free = setup.memory?.system?.available_mb;
        if (total) {
            grid.append(deviceCard('System memory', `${(total / 1024).toFixed(1)} GB installed`,
                free ? `${(free / 1024).toFixed(1)} GB available now` : ''));
        }
        if (!grid.children.length) grid.append(deviceCard('CPU', 'OpenVINO fallback', 'No accelerator reported'));
        stage.append(grid);

        const note = el('p', 'setup-note');
        note.style.marginTop = 'var(--sp-4)';
        note.textContent = setup.ollama?.available
            ? `Ollama is running with ${setup.ollama.models.length} installed model${setup.ollama.models.length === 1 ? '' : 's'}; locally can use it without copying the weights.`
            : 'All recommendations below run locally. Downloads happen only after the final confirmation.';
        stage.append(note);
    }

    function choice(item, group, checked, onChange, extra = {}) {
        const label = el('label', 'setup-choice');
        label.dataset.installed = String(Boolean(item.installed));
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = group;
        input.value = item.id || item.name;
        input.checked = checked;
        input.addEventListener('change', onChange);
        const detail = item.detail || extra.detail || '';
        label.append(input, el('strong', '', item.name), el('span', '', detail),
            el('span', 'setup-choice-meta', extra.meta || gb(item.size_gb)));
        return label;
    }

    function renderAssistant() {
        const kinds = new Set((setup.devices || []).map(device => device.kind));
        const local = [...catalog.values()].filter(item => item.component === 'assistant'
            && (item.devices || []).some(device => kinds.has(device) || device === 'CPU'));
        const localTitle = el('div', 'setup-section-title', 'OpenVINO · on this machine');
        const localGrid = el('div', 'setup-choice-grid');
        for (const item of local) {
            const suffix = item.recommended ? ' · recommended' : '';
            localGrid.append(choice(item, 'setup-assistant', backend === 'openvino' && assistant === item.id, () => {
                backend = 'openvino';
                assistant = item.id;
            }, { meta: `${gb(item.size_gb)}${suffix}` }));
        }
        stage.append(localTitle, localGrid);
        if (setup.recommended?.assistant_reason) {
            const reason = el('p', 'setup-note', setup.recommended.assistant_reason);
            reason.style.marginTop = 'var(--sp-3)';
            stage.append(reason);
        }

        if (setup.ollama?.models?.length) {
            stage.append(el('div', 'setup-section-title', 'Ollama · already installed'));
            const ollamaGrid = el('div', 'setup-choice-grid');
            for (const model of setup.ollama.models) {
                const item = { id: model.name, name: model.name, installed: true,
                    detail: 'Use the existing Ollama model through locally’s API' };
                const size = model.size ? gb(model.size / 1024 ** 3) : 'installed';
                ollamaGrid.append(choice(item, 'setup-assistant', backend === 'ollama' && ollamaModel === model.name, () => {
                    backend = 'ollama';
                    ollamaModel = model.name;
                }, { meta: size }));
            }
            stage.append(ollamaGrid);
        }
    }

    function toggleRow(item, checked, onChange, inputType = 'checkbox', group = '') {
        const label = el('label', 'setup-toggle-row');
        const copy = el('span', 'setup-toggle-copy');
        copy.append(el('strong', '', item.name),
            el('span', '', `${item.detail} · ${item.installed ? 'installed' : gb(item.size_gb)}`));
        const input = document.createElement('input');
        input.type = inputType;
        if (group) input.name = group;
        input.checked = checked;
        input.addEventListener('change', onChange);
        label.append(copy, input);
        return label;
    }

    function renderVoice() {
        stage.append(el('div', 'setup-section-title', 'Speech recognition'));
        const sttList = el('div', 'setup-toggle-list');
        for (const item of [...catalog.values()].filter(value => value.component === 'stt')) {
            sttList.append(toggleRow(item, stt === item.id, () => { stt = item.id; }, 'radio', 'setup-stt'));
        }
        stage.append(sttList, el('div', 'setup-section-title', 'Conversation'));
        const voiceList = el('div', 'setup-toggle-list');
        for (const component of ['tts', 'vad', 'speaker']) {
            const item = [...catalog.values()].find(value => value.component === component);
            if (!item) continue;
            voiceList.append(toggleRow(item, voice.has(item.id), event => {
                if (event.currentTarget.checked) voice.add(item.id);
                else voice.delete(item.id);
            }));
        }
        stage.append(voiceList);
        const note = el('p', 'setup-note');
        note.style.marginTop = 'var(--sp-4)';
        note.textContent = 'Speaker identity is optional, but it is what prevents another person or a television from taking your turn. You enroll your voice after setup.';
        stage.append(note);
    }

    function renderWeb() {
        const list = el('div', 'setup-toggle-list');
        const reader = { name: 'Read any public URL', detail: 'Built in · extracts the useful page as Markdown', installed: true };
        const readerRow = toggleRow(reader, true, () => {});
        readerRow.querySelector('input').disabled = true;
        list.append(readerRow);
        const search = catalog.get('searxng');
        if (search) {
            list.append(toggleRow(search, searxng, event => { searxng = event.currentTarget.checked; }));
        }
        const updates = {
            name: 'Check for project updates',
            detail: 'Once a day · GitHub and npm · never installs automatically',
            installed: true,
        };
        list.append(toggleRow(updates, checkUpdates, event => {
            checkUpdates = event.currentTarget.checked;
        }));
        stage.append(list);
        const note = el('p', 'setup-note');
        note.style.marginTop = 'var(--sp-4)';
        note.textContent = 'URL reading fetches only pages you ask for. SearXNG is a separate open-source metasearch service, installed locally and updated independently.';
        stage.append(note);
    }

    function reviewRow(name, detail) {
        const row = el('div', 'setup-review-row');
        row.append(el('strong', '', name), el('span', '', detail));
        return row;
    }

    function renderReview() {
        if (finished) {
            stage.append(deviceCard('Ready', 'Your assistant stack is configured.',
                'You can change every choice later in Settings'));
            return;
        }
        const list = el('div', 'setup-review-list');
        if (backend === 'ollama') {
            list.append(reviewRow('Assistant', `${ollamaModel || 'No model selected'} · Ollama`));
        } else {
            const item = catalog.get(assistant);
            list.append(reviewRow('Assistant', item ? `${item.name} · ${item.installed ? 'installed' : gb(item.size_gb)}` : 'Choose a model'));
        }
        const voiceNames = [stt, ...voice].map(id => catalog.get(id)?.name).filter(Boolean);
        list.append(reviewRow('Voice', voiceNames.length ? voiceNames.join(' · ') : 'Text only'));
        list.append(reviewRow('Web', searxng ? 'URL reader · SearXNG search' : 'URL reader only'));
        list.append(reviewRow('Updates', checkUpdates ? 'Daily notice only' : 'Off'));
        stage.append(list);

        const missing = selectedIds().map(id => catalog.get(id)).filter(item => item && !item.installed);
        const total = missing.reduce((sum, item) => sum + Number(item.size_gb || 0), 0);
        const note = el('p', 'setup-note');
        note.style.marginTop = 'var(--sp-4)';
        note.textContent = missing.length
            ? `${missing.length} missing component${missing.length === 1 ? '' : 's'} will download (about ${gb(total)}). Installed models are reused.`
            : 'Everything selected is already on this machine. No model download is needed.';
        stage.append(note);
    }

    const renderers = [renderHardware, renderAssistant, renderVoice, renderWeb, renderReview];

    function renderDots() {
        dots.replaceChildren();
        steps.forEach((item, index) => {
            const dot = el('button', 'setup-dot');
            dot.type = 'button';
            dot.setAttribute('aria-label', `Step ${index + 1}: ${item.title}`);
            dot.setAttribute('aria-current', index === step ? 'step' : 'false');
            dot.addEventListener('click', () => {
                if (busy) return;
                step = index;
                render();
            });
            dots.append(dot);
        });
    }

    function render() {
        const copy = finished ? { kicker: 'READY', title: 'Your assistant is ready',
            lede: 'locally will remember this setup on the next launch.' } : steps[step];
        kicker.textContent = copy.kicker;
        title.textContent = copy.title;
        lede.textContent = copy.lede;
        stage.replaceChildren();
        renderers[step]();
        renderDots();
        back.hidden = step === 0 || finished;
        close.hidden = Boolean(setup.needs_assistant && !finished);
        next.textContent = finished ? 'Start chatting' : step === steps.length - 1 ? 'Install & finish' : 'Continue';
        next.disabled = busy;
        status.textContent = '';
        // Announce the transition, not the contents: the step's cards, options
        // and hints are read on demand, once, by navigating them.
        stepStatus.textContent = finished
            ? copy.title
            : `Step ${step + 1} of ${steps.length}: ${copy.title}`;
    }

    function showInstallProgress(ids) {
        stage.replaceChildren();
        const progress = el('div', 'setup-progress');
        progress.setAttribute('aria-label', 'Installation progress');
        progress.append(el('div', 'setup-progress-fill'));
        const list = el('div', 'setup-install-list');
        for (const id of ids) {
            const item = catalog.get(id);
            const row = el('div', 'setup-install-item', `${item?.name || id} · waiting`);
            row.dataset.item = id;
            row.dataset.state = 'queued';
            list.append(row);
        }
        stage.append(progress, list);
        return { fill: progress.firstElementChild, list };
    }

    function paintJob(ui, job) {
        const states = job.items || {};
        let done = 0;
        for (const [id, stateValue] of Object.entries(states)) {
            const row = ui.list.querySelector(`[data-item="${CSS.escape(id)}"]`);
            if (!row) continue;
            const item = catalog.get(id);
            const stateText = stateValue === 'downloading' ? 'downloading'
                : stateValue === 'installed' ? 'ready'
                : stateValue === 'error' ? 'failed' : 'waiting';
            row.textContent = `${item?.name || id} · ${stateText}`;
            row.dataset.state = stateValue;
            if (stateValue === 'installed') done += 1;
        }
        ui.fill.style.width = `${states && Object.keys(states).length ? done / Object.keys(states).length * 100 : 0}%`;
        status.textContent = job.current ? `Installing ${catalog.get(job.current)?.name || job.current}` : '';
    }

    const wait = ms => new Promise(resolve => window.setTimeout(resolve, ms));

    async function installAndFinish() {
        if (backend === 'openvino' && !assistant) {
            status.textContent = 'Choose an assistant model';
            return;
        }
        if (backend === 'ollama' && !ollamaModel) {
            status.textContent = 'Choose an Ollama model';
            return;
        }
        busy = true;
        next.disabled = true;
        back.disabled = true;
        close.hidden = true;
        const ids = selectedIds();
        const ui = showInstallProgress(ids);
        try {
            const missing = ids.filter(id => !catalog.get(id)?.installed);
            if (missing.length) {
                const response = await fetch('/v1/setup/install', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ items: missing }),
                });
                let job = await response.json();
                if (!response.ok) throw new Error(job.error?.message || 'Could not start installation');
                paintJob(ui, job);
                while (!['complete', 'error'].includes(job.status)) {
                    await wait(750);
                    const poll = await fetch(`/v1/setup/install/${encodeURIComponent(job.id)}`, { cache: 'no-store' });
                    job = await poll.json();
                    if (!poll.ok) throw new Error(job.error?.message || 'Installation status failed');
                    paintJob(ui, job);
                }
                if (job.status === 'error') throw new Error(job.error || 'Installation failed');
                for (const id of missing) catalog.get(id).installed = true;
            }

            status.textContent = 'Loading your local stack';
            const finishResponse = await fetch('/v1/setup/finish', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    backend, assistant, ollama_model: ollamaModel,
                    device: catalog.get(assistant)?.devices?.find(device =>
                        (setup.devices || []).some(found => found.kind === device)),
                    voice: [stt, ...voice].filter(Boolean), searxng,
                    check_updates: checkUpdates,
                }),
            });
            const result = await finishResponse.json();
            if (!finishResponse.ok) throw new Error(result.error?.message || 'Could not finish setup');

            if (backend === 'openvino' && !result.assistant.loaded) {
                status.textContent = 'Compiling the assistant for this device';
                const loadResponse = await fetch('/v1/models/load', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: result.assistant.path, device: result.assistant.device }),
                });
                const loaded = await loadResponse.json();
                if (!loadResponse.ok) throw new Error(loaded.error?.message || 'The assistant could not load');
            }

            localStorage.setItem('locally-onboarding-complete', '1');
            finished = true;
            status.textContent = '';
            render();
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            // One announcement: `status` is role="status" and speaks; the line
            // in the stage is the silent visual twin, styled by class.
            status.textContent = message;
            stage.append(el('p', 'setup-note is-error', message));
        } finally {
            busy = false;
            next.disabled = false;
            back.disabled = false;
        }
    }

    // --- focus containment ---------------------------------------------------
    // The dialog declared aria-modal="true" and nothing kept Tab inside it, so
    // assistive tech was told the app behind was inert when it was not -- and on
    // first run the dialog cannot be dismissed, which means a keyboard user could
    // Tab into an app they had no way back out of. `inert` makes the claim true;
    // the Tab handler is the same cycle palette.js and swap.js already use.
    const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),'
        + 'select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

    let returnFocusTo = null;

    const focusablesIn = root =>
        [...root.querySelectorAll(FOCUSABLE)].filter(node => node.offsetParent !== null);

    // Capture, so it runs before the app's own Tab handling -- the Escape
    // handler below is registered the same way for the same reason.
    function onTrapKey(event) {
        if (event.key !== 'Tab') return;
        const items = focusablesIn(dialog);
        if (!items.length) { event.preventDefault(); return; }
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement;
        if (event.shiftKey && (active === first || !dialog.contains(active))) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && (active === last || !dialog.contains(active))) {
            event.preventDefault();
            first.focus();
        }
    }

    // "Everything except this subtree" is a walk UP from the dialog, inerting
    // each ancestor's other children. Inerting only `document.body.children`
    // looks right and is wrong here: the shell is nested inside `.app-column`,
    // so that version inerted the dialog along with the app and focus could not
    // enter it at all — `dialog.focus()` silently did nothing.
    const inertOutside = value => {
        for (let node = shell; node && node !== document.body; node = node.parentElement) {
            for (const sibling of node.parentElement.children) {
                if (sibling !== node) sibling.inert = value;
            }
        }
    };

    // The app appends while setup is open -- the update card, toasts, the memory
    // HUD. Inerting once left those live behind a dialog claiming aria-modal,
    // which is the exact hole this code exists to close, so anything added
    // outside the dialog's own ancestry is inerted as it arrives.
    const watchNewSiblings = new MutationObserver(records => {
        for (const record of records) {
            for (const node of record.addedNodes) {
                // Anything on the path to the dialog, and the dialog's own
                // content, must stay live -- render() replaces the stage on
                // every step, and inerting that would empty the dialog of
                // anything focusable.
                if (node.nodeType !== 1) continue;
                if (node === shell || node.contains(shell) || shell.contains(node)) continue;
                node.inert = true;
            }
        }
    });

    // Watch each level of the dialog's ancestry, so a new sibling at any depth
    // of that path is caught without observing the whole document.
    const observePath = () => {
        for (let node = shell; node && node !== document.body; node = node.parentElement) {
            watchNewSiblings.observe(node.parentElement, { childList: true });
        }
    };

    function setOpen(open) {
        shell.hidden = !open;
        shell.setAttribute('aria-hidden', String(!open));
        document.body.dataset.setupOpen = open ? '1' : '0';
        if (open) {
            // Read the outgoing focus BEFORE inerting, or the element holding it
            // is already unfocusable and there is nothing to return to.
            returnFocusTo = document.activeElement;
            inertOutside(true);
            observePath();
            document.addEventListener('keydown', onTrapKey, true);
            // Focus moves after the inerting, not before: inerting the button
            // that opened setup drops focus to <body>, so a focus() scheduled
            // ahead of that is simply undone. The dialog, not Continue --
            // opening a step with the primary action focused puts a stray Enter
            // one keystroke from advancing past content nobody has read.
            dialog.focus();
        } else {
            watchNewSiblings.disconnect();
            inertOutside(false);
            document.removeEventListener('keydown', onTrapKey, true);
            returnFocusTo?.focus?.();
            returnFocusTo = null;
        }
    }

    async function openSetup(force = false) {
        setOpen(true);
        busy = true;
        next.disabled = true;
        stage.replaceChildren(deviceCard('Inspecting this device', 'Reading local OpenVINO devices and installed models.', 'No data leaves this machine'));
        try {
            const response = await fetch('/v1/setup', { cache: 'no-store' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || 'Setup information is unavailable');
            setup = data;
            catalog = new Map((data.catalog || []).map(item => [item.id, item]));
            initializeSelections();
            step = 0;
            finished = false;
            render();
        } catch (error) {
            title.textContent = 'Setup could not start';
            lede.textContent = error instanceof Error ? error.message : String(error);
            stage.replaceChildren();
            close.hidden = false;
        } finally {
            busy = false;
            next.disabled = false;
        }
        if (!force && setup && !setup.first_run && !setup.needs_assistant
                && localStorage.getItem('locally-onboarding-complete') === '1') {
            setOpen(false);
        }
    }

    back.addEventListener('click', () => {
        if (!busy && step > 0) { step -= 1; render(); }
    });
    next.addEventListener('click', async () => {
        if (busy) return;
        if (finished) {
            setOpen(false);
            window.dispatchEvent(new Event('focus'));
        } else if (step < steps.length - 1) {
            step += 1;
            render();
        } else {
            await installAndFinish();
        }
    });
    close.addEventListener('click', () => {
        if (!busy && (!setup?.needs_assistant || finished)) setOpen(false);
    });
    openButton.addEventListener('click', () => openSetup(true));
    document.addEventListener('keydown', event => {
        if (shell.hidden || event.key !== 'Escape') return;
        if (!busy && (!setup?.needs_assistant || finished)) setOpen(false);
    }, true);

    openSetup(false);
})();
