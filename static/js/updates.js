// Optional update notice. The server returns enabled:false unless the user
// explicitly started locally with --check-updates, so merely loading this file
// never makes the app phone home.
(() => {
    const labels = {
        locally: 'locally',
        odysseus: 'Odysseus',
        searxng: 'SearXNG',
        opencode: 'OpenCode',
    };

    const shortVersion = (name, value) => {
        if (!value) return '';
        return name === 'locally' ? value.slice(0, 7) : value;
    };

    function showNotice(sources) {
        const available = Object.entries(sources)
            .filter(([, value]) => value?.update_available);
        if (!available.length) return;

        const fingerprint = available
            .map(([name, value]) => `${name}:${value.latest}`)
            .sort().join('|');
        if (localStorage.getItem('locally-dismissed-updates') === fingerprint) return;

        const notice = document.createElement('aside');
        notice.className = 'update-notice';
        notice.setAttribute('role', 'status');
        notice.setAttribute('aria-label', 'Updates available');

        const copy = document.createElement('div');
        copy.className = 'update-copy';
        const title = document.createElement('strong');
        title.textContent = available.length === 1
            ? 'An update is available'
            : `${available.length} updates are available`;
        const detail = document.createElement('span');
        detail.textContent = available.map(([name, value]) =>
            `${labels[name] || name} ${shortVersion(name, value.latest)}`
        ).join(' · ');
        copy.append(title, detail);

        const links = document.createElement('div');
        links.className = 'update-links';
        for (const [name, value] of available) {
            const link = document.createElement('a');
            link.href = value.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = labels[name] || name;
            links.append(link);
        }

        const dismiss = document.createElement('button');
        dismiss.className = 'update-dismiss';
        dismiss.type = 'button';
        dismiss.setAttribute('aria-label', 'Dismiss update notice');
        dismiss.textContent = '×';
        dismiss.addEventListener('click', () => {
            localStorage.setItem('locally-dismissed-updates', fingerprint);
            notice.remove();
        });

        notice.append(copy, links, dismiss);
        document.body.append(notice);
    }

    async function check(attempt = 0) {
        try {
            const response = await fetch('/v1/updates', { cache: 'no-store' });
            if (!response.ok) return;
            const data = await response.json();
            if (!data.enabled) return;
            if (data.refreshing && attempt < 8) {
                window.setTimeout(() => check(attempt + 1), 1000);
                return;
            }
            showNotice(data.sources || {});
        } catch {
            // Offline is a normal operating mode. No error chrome, no retry
            // storm, and nothing delays the rest of the app.
        }
    }

    check();
})();
