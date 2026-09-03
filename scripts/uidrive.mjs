// Run a script against the web UI in a real browser, from the command line.
//
// css-oracle.js says "paste this file into the console on the running app".
// That works once and is not a test: it needs a human, a visible pane and a
// steady hand at three viewport widths. Worse, a hidden pane is not merely
// inconvenient -- `document.hidden` is true there, requestAnimationFrame
// never fires, and every measurement that waits for a frame hangs instead of
// returning. The rig has to own the browser.
//
// This drives the Chromium that ships with WebView2/Edge over the DevTools
// protocol with no npm dependency at all: Node 21+ has a global WebSocket and
// CDP is JSON over one socket. A headless page is a VISIBLE page as far as the
// page can tell, so rAF runs, fonts load, and IntersectionObserver fires.
//
//   node scripts/uidrive.mjs --url http://127.0.0.1:8778/ \
//        --width 1280 --height 800 --script scratchpad/probe.js
//
// --script is evaluated as the body of an async function in the page and its
// return value is printed as JSON. --oracle evaluates scripts/css-oracle.js
// first, so the probe can call cssOracle.
//
// Anything set with --set k=v is applied through
// Page.addScriptToEvaluateOnNewDocument, i.e. before the app's first line
// runs. Setting locally-onboarding-complete after load is too late: the setup
// dialog is already open and the shell is already behind a modal.

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CORES = [
    'C:/Program Files (x86)/Microsoft/EdgeCore',
    'C:/Program Files/Microsoft/EdgeCore',
];
const DIRECT = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
];

// WebView2 is what the desktop shell renders in, so measuring in EdgeCore is
// measuring the engine the app actually ships on -- preferred over a Chrome
// that may be a different major version.
function findBrowser() {
    if (process.env.LOCALLY_BROWSER) return process.env.LOCALLY_BROWSER;
    for (const root of CORES) {
        if (!existsSync(root)) continue;
        for (const version of readdirSync(root).sort().reverse()) {
            const exe = join(root, version, 'msedge.exe');
            if (existsSync(exe)) return exe;
        }
    }
    for (const exe of DIRECT) if (existsSync(exe)) return exe;
    throw new Error('No Chromium found. Set LOCALLY_BROWSER to a chrome/msedge path.');
}

function parseArgs() {
    const out = { width: 1280, height: 800, set: [], port: 0, wait: 400 };
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i++) {
        const key = argv[i].replace(/^--/, '');
        if (key === 'set') out.set.push(argv[++i]);
        else if (key === 'oracle') out.oracle = true;
        else {
            const value = argv[++i];
            out[key] = /^\d+$/.test(value || '') ? Number(value) : value;
        }
    }
    return out;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function devtoolsUrl(port) {
    for (let i = 0; i < 150; i++) {
        try {
            const r = await fetch(`http://127.0.0.1:${port}/json/version`);
            if (r.ok) return (await r.json()).webSocketDebuggerUrl;
        } catch { /* not listening yet */ }
        await sleep(100);
    }
    throw new Error('Browser never opened its debugging port');
}

class Cdp {
    constructor(ws) {
        this.ws = ws;
        this.id = 0;
        this.waiting = new Map();
        this.events = [];
        ws.addEventListener('message', ev => {
            const msg = JSON.parse(ev.data);
            if (msg.id && this.waiting.has(msg.id)) {
                const { resolve, reject } = this.waiting.get(msg.id);
                this.waiting.delete(msg.id);
                if (msg.error) reject(new Error(JSON.stringify(msg.error)));
                else resolve(msg.result);
            } else if (msg.method) {
                this.events.push(msg);
            }
        });
    }
    send(method, params = {}, sessionId) {
        const id = ++this.id;
        const payload = { id, method, params };
        if (sessionId) payload.sessionId = sessionId;
        this.ws.send(JSON.stringify(payload));
        return new Promise((resolve, reject) => this.waiting.set(id, { resolve, reject }));
    }
    async until(method, timeoutMs = 30000) {
        const deadline = Date.now() + timeoutMs;
        while (Date.now() < deadline) {
            const hit = this.events.findIndex(e => e.method === method);
            if (hit >= 0) return this.events.splice(hit, 1)[0];
            await sleep(50);
        }
        throw new Error(`Timed out waiting for ${method}`);
    }
}

async function main() {
    const args = parseArgs();
    if (!args.url || !args.script) {
        console.error('usage: node scripts/uidrive.mjs --url URL --script FILE '
            + '[--width N] [--height N] [--set k=v] [--oracle]');
        process.exit(2);
    }
    const exe = findBrowser();
    const profile = mkdtempSync(join(tmpdir(), 'locally-uidrive-'));
    const port = args.port || 9222 + (process.pid % 500);
    const child = spawn(exe, [
        '--headless=new', `--remote-debugging-port=${port}`,
        `--user-data-dir=${profile}`, `--window-size=${args.width},${args.height}`,
        '--no-first-run', '--no-default-browser-check', '--disable-extensions',
        '--hide-scrollbars', '--force-device-scale-factor=1',
        // performance.memory is bucketed to 100 KB without this, which is
        // coarser than most of what is worth measuring in a page.
        '--enable-precise-memory-info', 'about:blank',
    ], { stdio: 'ignore' });

    let code = 0;
    try {
        const ws = new WebSocket(await devtoolsUrl(port));
        await new Promise((res, rej) => {
            ws.addEventListener('open', res);
            ws.addEventListener('error', rej);
        });
        const cdp = new Cdp(ws);
        const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
        const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
        const call = (m, p) => cdp.send(m, p, sessionId);

        await call('Page.enable');
        await call('Runtime.enable');
        // A page that throws during boot can still answer a probe, and the
        // probe will happily report success. Collect what the console saw so
        // "it works" and "it did not log an error" are the same claim.
        await call('Log.enable');
        // A real layout viewport, owned by the rig. This is what the oracle's
        // "capture() refuses to run at zero width" is guarding against.
        await call('Emulation.setDeviceMetricsOverride', {
            width: Number(args.width), height: Number(args.height),
            deviceScaleFactor: 1, mobile: false,
        });

        for (const pair of args.set) {
            const [k, ...rest] = pair.split('=');
            await call('Page.addScriptToEvaluateOnNewDocument', {
                source: `try{localStorage.setItem(${JSON.stringify(k)},`
                    + `${JSON.stringify(rest.join('='))})}catch(e){}`,
            });
        }

        // --arg is how a probe is run more than once against different stopping
        // points without editing it. It lands before any app code runs.
        if (args.arg !== undefined) {
            await call('Page.addScriptToEvaluateOnNewDocument', {
                source: `window.__arg = ${JSON.stringify(String(args.arg))};`,
            });
        }

        await call('Page.navigate', { url: args.url });
        await cdp.until('Page.loadEventFired');
        // Boot is modules plus a frame, and the oracle's own rule is to wait
        // for fonts before measuring anything whose value is a width.
        await call('Runtime.evaluate', {
            expression: 'new Promise(r=>setTimeout(()=>document.fonts.ready.then('
                + '()=>requestAnimationFrame(()=>requestAnimationFrame(r)))'
                + `, ${Number(args.wait)}))`,
            awaitPromise: true,
        });

        if (args.oracle) {
            await call('Runtime.evaluate', { expression: readFileSync('scripts/css-oracle.js', 'utf8') });
        }

        const body = readFileSync(args.script, 'utf8');
        const res = await call('Runtime.evaluate', {
            expression: `(async () => { ${body} })()`,
            awaitPromise: true, returnByValue: true,
        });
        // After the probe, not before: a screenshot is worth taking of the
        // state the probe left the page in, which is the state being argued
        // about. Written as a file so it can go in a report.
        if (args.shot) {
            const { data } = await call('Page.captureScreenshot', { format: 'png' });
            writeFileSync(args.shot, Buffer.from(data, 'base64'));
        }

        if (res.exceptionDetails) {
            console.error(res.exceptionDetails.exception?.description
                || JSON.stringify(res.exceptionDetails));
            code = 1;
        } else {
            const value = res.result.value;
            // querySelectorAll('*').length counts elements the page can see.
            // The renderer's own counter includes text nodes and detached
            // trees, and DOM nodes do not live on the JS heap -- which is why
            // performance.memory barely moves when thousands of them go.
            if (value && typeof value === 'object' && !Array.isArray(value)) {
                try { value.__domCounters = await call('Memory.getDOMCounters'); }
                catch { /* Not every build exposes the Memory domain. */ }
                // Failed asset fetches are the rig's own stubs saying 404 and
                // are not the page's problem; script errors are.
                value.__errors = cdp.events
                    .filter(e => e.method === 'Log.entryAdded'
                        && e.params.entry.level === 'error'
                        && e.params.entry.source !== 'network')
                    .map(e => e.params.entry.text);
            }
            console.log(JSON.stringify(value, null, 2));
        }
        ws.close();
    } catch (err) {
        console.error(String((err && err.stack) || err));
        code = 1;
    } finally {
        child.kill();
        await sleep(300);
        try { rmSync(profile, { recursive: true, force: true }); } catch { /* still locked */ }
    }
    process.exit(code);
}

main();
