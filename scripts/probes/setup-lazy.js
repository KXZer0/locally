// The setup overlay is now cloned from a <template> on demand. That saves 29
// elements on every returning user's boot and is worthless if it costs the
// first-run user their only way in, so this checks both halves against the
// same page: absent when it should be, and fully working when it should not.
//
// Run twice:
//   --arg returning   with locally-onboarding-complete=1 set
//   --arg firstrun    without it
//
// The first-run pass is the one that matters. It waits for the overlay to
// appear rather than assuming it is synchronous: the boot path now asks
// /v1/setup BEFORE building anything, so the dialog arrives a round trip late.

const mode = window.__arg || 'returning';
const until = async (fn, ms = 4000) => {
    const t0 = performance.now();
    while (performance.now() - t0 < ms) {
        const v = fn();
        if (v) return v;
        await new Promise(r => setTimeout(r, 50));
    }
    return null;
};

const template = document.getElementById('setup-template');
const out = {
    mode,
    templateExists: !!template,
    templateElements: template ? template.content.querySelectorAll('*').length : 0,
};

if (mode === 'returning') {
    // Give the boot probe its round trip, then confirm nothing was built.
    await new Promise(r => setTimeout(r, 1200));
    const shell = document.getElementById('setup-shell');
    out.shellInDocument = !!shell;
    out.bodyElements = document.querySelectorAll('body *').length;
    // The button that opens it must still work, or "not built" becomes
    // "unreachable".
    document.getElementById('setup-open-btn')?.click();
    const opened = await until(() => document.getElementById('setup-shell'));
    out.opensOnDemand = !!opened;
    out.openedHidden = opened ? opened.hidden : null;
    out.dialogFocusable = !!document.getElementById('setup-dialog');
} else {
    const shell = await until(() => {
        const el = document.getElementById('setup-shell');
        return el && !el.hidden ? el : null;
    });
    out.shellInDocument = !!shell;
    out.visible = !!shell && !shell.hidden;
    // Everything the dialog needs in order to be usable at all.
    for (const id of ['setup-dialog', 'setup-stage', 'setup-next', 'setup-back',
                      'setup-dots', 'setup-status', 'setup-step-status']) {
        out[id] = !!document.getElementById(id);
    }
    // The step actually rendered, rather than an empty shell.
    const stage = document.getElementById('setup-stage');
    out.stageHasContent = !!stage && stage.children.length > 0;
    out.title = document.getElementById('setup-title')?.textContent || null;
    out.dots = document.getElementById('setup-dots')?.children.length ?? 0;
    // And the wiring: Continue must advance a step.
    const before = out.title;
    document.getElementById('setup-next')?.click();
    await new Promise(r => setTimeout(r, 250));
    out.titleAfterContinue = document.getElementById('setup-title')?.textContent || null;
    out.continueAdvances = out.titleAfterContinue !== before;
    // The focus trap from §2.1a has to survive being cloned.
    out.dialogHasTabindex =
        document.getElementById('setup-dialog')?.getAttribute('tabindex') === '-1';
    out.appInert = document.querySelector('.app-column')?.inert ?? null;
}

return out;
