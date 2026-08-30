// locally — TeX → MathML
//
// Why hand-rolled: this UI is offline-first and self-hosted (same reason the
// fonts live in static/fonts/), so a CDN'd KaTeX is out, and vendoring KaTeX
// costs ~1 MB of JS plus its own font files to cover a corner of TeX no chat
// model ever emits. Browsers render MathML natively (Chrome 109+, Edge,
// Firefox, Safari 14+), so the entire job is TeX → MathML and the math font
// is the system's.
//
// Scope is deliberately "what an LLM writes": fractions, roots, scripts, big
// operators with limits, greek, relations, accents, \left…\right, and the
// matrix / cases / aligned environments. An unknown command degrades to its
// own name rather than throwing, and a hard failure is caught upstream in
// renderMath(), which falls back to showing the raw TeX.

(function () {
'use strict';

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// --- Symbol tables ---

// Lowercase greek is italic in TeX, uppercase upright — MathML gets that right
// for <mi> only if we say so, since a single-char <mi> is italic by default.
const GREEK_LOWER = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ϵ', varepsilon: 'ε',
    zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ',
    lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', omicron: 'ο', pi: 'π', varpi: 'ϖ',
    rho: 'ρ', varrho: 'ϱ', sigma: 'σ', varsigma: 'ς', tau: 'τ', upsilon: 'υ',
    phi: 'ϕ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
};
const GREEK_UPPER = {
    Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π',
    Sigma: 'Σ', Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
};

const OPS = {
    times: '×', div: '÷', pm: '±', mp: '∓', cdot: '⋅', ast: '∗', star: '⋆',
    circ: '∘', bullet: '∙', amalg: '⨿', wr: '≀', dagger: '†', ddagger: '‡',
    leq: '≤', le: '≤', geq: '≥', ge: '≥', neq: '≠', ne: '≠', equiv: '≡',
    approx: '≈', approxeq: '≊', cong: '≅', sim: '∼', simeq: '≃', propto: '∝',
    ll: '≪', gg: '≫', prec: '≺', succ: '≻', preceq: '⪯', succeq: '⪰',
    doteq: '≐', asymp: '≍', bowtie: '⋈', lesssim: '≲', gtrsim: '≳',
    subset: '⊂', supset: '⊃', subseteq: '⊆', supseteq: '⊇', sqsubseteq: '⊑',
    in: '∈', notin: '∉', ni: '∋', cup: '∪', cap: '∩', uplus: '⊎',
    sqcup: '⊔', sqcap: '⊓', setminus: '∖', emptyset: '∅', varnothing: '∅',
    forall: '∀', exists: '∃', nexists: '∄', neg: '¬', lnot: '¬',
    land: '∧', lor: '∨', wedge: '∧', vee: '∨', top: '⊤', bot: '⊥',
    to: '→', gets: '←', rightarrow: '→', leftarrow: '←', leftrightarrow: '↔',
    Rightarrow: '⇒', Leftarrow: '⇐', Leftrightarrow: '⇔', mapsto: '↦',
    implies: '⟹', impliedby: '⟸', iff: '⟺', longrightarrow: '⟶',
    longleftarrow: '⟵', longleftrightarrow: '⟷', hookrightarrow: '↪',
    uparrow: '↑', downarrow: '↓', updownarrow: '↕', nearrow: '↗', searrow: '↘',
    infty: '∞', partial: '∂', nabla: '∇', hbar: 'ℏ', ell: 'ℓ', wp: '℘',
    Re: 'ℜ', Im: 'ℑ', aleph: 'ℵ', imath: 'ı', jmath: 'ȷ',
    degree: '°', angle: '∠', measuredangle: '∡', perp: '⊥', parallel: '∥',
    therefore: '∴', because: '∵', triangleq: '≜', propto2: '∝',
    ldots: '…', cdots: '⋯', vdots: '⋮', ddots: '⋱', dots: '…', dotsc: '…',
    prime: '′', oplus: '⊕', ominus: '⊖', otimes: '⊗', oslash: '⊘', odot: '⊙',
    models: '⊨', vdash: '⊢', dashv: '⊣', mid: '∣', nmid: '∤', parallel2: '∥',
    lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉', langle: '⟨', rangle: '⟩',
    lbrace: '{', rbrace: '}', lbrack: '[', rbrack: ']', backslash: '\\',
    surd: '√', checkmark: '✓', square: '□', blacksquare: '■', triangle: '△',
    diamond: '⋄', clubsuit: '♣', heartsuit: '♡', flat: '♭', sharp: '♯',
    S: '§', P: '¶', copyright: '©', pounds: '£', euro: '€',
};

// Big operators. The flag says whether TeX puts limits above/below (∑) or
// beside (∫) — that distinction is the whole reason MathML has both
// <munderover> and <msubsup>.
const BIG = {
    sum: ['∑', true], prod: ['∏', true], coprod: ['∐', true],
    bigcup: ['⋃', true], bigcap: ['⋂', true], bigsqcup: ['⨆', true],
    bigoplus: ['⨁', true], bigotimes: ['⨂', true], bigodot: ['⨀', true],
    bigvee: ['⋁', true], bigwedge: ['⋀', true], biguplus: ['⨄', true],
    int: ['∫', false], iint: ['∬', false], iiint: ['∭', false],
    oint: ['∮', false], oiint: ['∯', false],
};

// Upright multi-letter operators that take under/over limits.
const NAMED_LIMITS = {
    lim: 'lim', limsup: 'lim sup', liminf: 'lim inf', max: 'max', min: 'min',
    sup: 'sup', inf: 'inf', argmax: 'arg max', argmin: 'arg min',
};

// Upright function names; scripts stay beside them.
const FUNCS = new Set([
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'coth', 'log', 'ln', 'lg', 'exp', 'det', 'dim',
    'gcd', 'deg', 'ker', 'hom', 'Pr', 'mod', 'bmod', 'arg',
]);

const ACCENTS = {
    hat: '^', widehat: '^', check: 'ˇ', tilde: '~', widetilde: '~',
    acute: '´', grave: '`', dot: '˙', ddot: '¨', breve: '˘', ring: '˚',
    bar: '‾', overline: '‾', vec: '→', overrightarrow: '→', overleftarrow: '←',
};

const SPACES = {
    ',': '0.167em', ':': '0.222em', ';': '0.278em', '!': '-0.167em',
    ' ': '0.25em', quad: '1em', qquad: '2em', enspace: '0.5em',
    thinspace: '0.167em', medspace: '0.222em', thickspace: '0.278em',
    negthinspace: '-0.167em',
};

// Commands whose argument is literal text or plain symbols, not math to parse.
// Captured raw by the tokenizer so spacing inside \text{} survives.
const STYLED = new Set([
    'text', 'textrm', 'textbf', 'textit', 'texttt', 'textsf', 'mbox',
    'operatorname', 'mathrm', 'mathbf', 'mathbb', 'mathcal', 'mathfrak',
    'mathsf', 'mathtt', 'mathit', 'boldsymbol', 'bm',
]);

// Unicode alphabets: MathML Core dropped every mathvariant except "normal",
// so blackboard/script/fraktur have to be real characters.
const ALPHABETS = {
    mathbb: Array.from('𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ'),
    mathcal: Array.from('𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵'),
    mathfrak: Array.from('𝔄𝔅ℭ𝔇𝔈𝔉𝔊ℌℑ𝔍𝔎𝔏𝔐𝔑𝔒𝔓𝔔ℜ𝔖𝔗𝔘𝔙𝔚𝔛𝔜ℨ'),
};

const DELIMS = {
    '(': '(', ')': ')', '[': '[', ']': ']', '|': '|', '/': '/',
    '{': '{', '}': '}', '\\|': '‖', '.': '', 'lbrace': '{', 'rbrace': '}',
    langle: '⟨', rangle: '⟩', lfloor: '⌊', rfloor: '⌋',
    lceil: '⌈', rceil: '⌉', vert: '|', Vert: '‖', uparrow: '↑', downarrow: '↓',
};

const ENV_FENCES = {
    pmatrix: ['(', ')'], bmatrix: ['[', ']'], Bmatrix: ['{', '}'],
    vmatrix: ['|', '|'], Vmatrix: ['‖', '‖'], cases: ['{', ''],
    matrix: ['', ''], smallmatrix: ['', ''], array: ['', ''],
    aligned: ['', ''], align: ['', ''], alignat: ['', ''], split: ['', ''],
    gathered: ['', ''], gather: ['', ''], subarray: ['', ''],
};

// --- Tokenizer ---

function tokenize(src) {
    const t = [];
    const n = src.length;
    let i = 0;

    // Reads a balanced {...} as raw source. Used for \text and friends, and
    // for environment names, where re-tokenizing would lose what matters.
    const braced = () => {
        while (src[i] === ' ') i++;
        if (src[i] !== '{') return null;
        let depth = 1;
        i++;
        const start = i;
        while (i < n && depth > 0) {
            if (src[i] === '\\') { i += 2; continue; }
            if (src[i] === '{') depth++;
            else if (src[i] === '}') depth--;
            i++;
        }
        return src.slice(start, depth === 0 ? i - 1 : i);
    };

    while (i < n) {
        const c = src[i];

        if (c === '\\') {
            const m = /^\\([a-zA-Z]+\*?|[\s\S])/.exec(src.slice(i));
            if (!m) { i++; continue; }
            const name = m[1];
            i += m[0].length;
            if (/[a-zA-Z]/.test(name[0])) while (src[i] === ' ') i++;

            if (name === 'begin' || name === 'end') {
                t.push({ type: name, v: braced() || '' });
                continue;
            }
            if (STYLED.has(name)) {
                const body = braced();
                t.push({ type: 'styled', cmd: name, v: body == null ? '' : body });
                continue;
            }
            t.push({ type: 'cmd', v: name });
            continue;
        }

        if (c === '{' || c === '}') { t.push({ type: c }); i++; continue; }
        if (c === '^') { t.push({ type: 'sup' }); i++; continue; }
        if (c === '_') { t.push({ type: 'sub' }); i++; continue; }
        if (c === '&') { t.push({ type: 'amp' }); i++; continue; }
        if (/\s/.test(c)) { i++; continue; }

        const num = /^\d+(?:[.,]\d+)*/.exec(src.slice(i));
        if (num) { t.push({ type: 'num', v: num[0] }); i += num[0].length; continue; }
        if (/[a-zA-Z]/.test(c)) { t.push({ type: 'letter', v: c }); i++; continue; }

        t.push({ type: 'other', v: c });
        i++;
    }
    return t;
}

// --- Emitters ---

function wrap(list) {
    if (list.length === 1) return list[0];
    return `<mrow>${list.join('')}</mrow>`;
}

function opML(ch) {
    if (ch === '-') return '<mo>−</mo>';          // proper minus, not hyphen
    if (ch === "'") return '<mo>′</mo>';
    if (ch === '~') return '<mspace width="0.25em"/>';
    return `<mo>${esc(ch)}</mo>`;
}

function charML(ch, style) {
    if (/\d/.test(ch)) return `<mn>${esc(ch)}</mn>`;
    if (/[a-zA-Z]/.test(ch)) {
        return `<mi mathvariant="normal"${style ? ` style="${style}"` : ''}>${esc(ch)}</mi>`;
    }
    if (ch === ' ') return '<mspace width="0.25em"/>';
    return opML(ch);
}

// \text{…} and the font commands. Anything with a backslash or brace inside is
// handed back to the parser and styled as a whole; the common case (letters,
// digits, a space) is emitted character by character.
function styledML(cmd, raw) {
    if (cmd === 'text' || cmd === 'textrm' || cmd === 'mbox' || cmd === 'textbf'
        || cmd === 'textit' || cmd === 'texttt' || cmd === 'textsf') {
        const style = cmd === 'textbf' ? ' style="font-weight:600"'
            : cmd === 'textit' ? ' style="font-style:italic"'
            : cmd === 'texttt' ? ' style="font-family:var(--mono,monospace)"'
            : '';
        // Edge spaces are meaningful ("\text{ if }") and would collapse.
        const body = esc(raw).replace(/^ /, ' ').replace(/ $/, ' ');
        return `<mtext${style}>${body}</mtext>`;
    }
    if (cmd === 'operatorname') return `<mi>${esc(raw)}</mi>`;

    if (/[\\{}^_]/.test(raw)) {
        const inner = wrap(new Parser(tokenize(raw)).list(null));
        const style = cmd === 'mathbf' || cmd === 'boldsymbol' || cmd === 'bm'
            ? ' style="font-weight:700"' : '';
        return `<mrow${style}>${inner}</mrow>`;
    }

    const alphabet = ALPHABETS[cmd];
    let out = '';
    for (const ch of raw) {
        if (alphabet && ch >= 'A' && ch <= 'Z') {
            out += `<mi>${alphabet[ch.charCodeAt(0) - 65]}</mi>`;
        } else if (cmd === 'mathit') {
            out += /[a-zA-Z]/.test(ch) ? `<mi>${esc(ch)}</mi>` : charML(ch, '');
        } else if (cmd === 'mathbf' || cmd === 'boldsymbol' || cmd === 'bm') {
            out += charML(ch, 'font-weight:700');
        } else if (cmd === 'mathsf') {
            out += charML(ch, 'font-family:sans-serif');
        } else if (cmd === 'mathtt') {
            out += charML(ch, 'font-family:var(--mono,monospace)');
        } else {
            out += charML(ch, '');
        }
    }
    return out.length ? (raw.length > 1 ? `<mrow>${out}</mrow>` : out) : '<mrow></mrow>';
}

// --- Parser ---

function Parser(tokens) {
    this.t = tokens;
    this.i = 0;
}

Parser.prototype.peek = function () { return this.t[this.i]; };

// Parses atoms until a closing brace, the end of input, or one of `stop`
// (command names, plus the pseudo-names '&' and 'end' for table structure).
Parser.prototype.list = function (stop) {
    const out = [];
    while (this.i < this.t.length) {
        const tk = this.t[this.i];
        if (tk.type === '}') break;
        if (stop) {
            if (tk.type === 'amp' && stop.has('&')) break;
            if (tk.type === 'end' && stop.has('end')) break;
            if (tk.type === 'cmd' && stop.has(tk.v)) break;
        }
        const before = this.i;
        const a = this.atom();
        if (a !== null) out.push(this.scripts(a));
        if (this.i === before) this.i++;   // never spin on a token we can't use
    }
    return out;
};

// One argument: a braced group, or the single atom that follows.
Parser.prototype.arg = function () {
    const tk = this.t[this.i];
    if (!tk) return '<mrow></mrow>';
    if (tk.type === '{') {
        this.i++;
        const inner = this.list(null);
        if (this.t[this.i] && this.t[this.i].type === '}') this.i++;
        return inner.length ? wrap(inner) : '<mrow></mrow>';
    }
    const before = this.i;
    const a = this.atom();
    if (this.i === before) this.i++;
    return a ? a.ml : '<mrow></mrow>';
};

// Optional [ … ] argument (\sqrt[3]{x}, \\[6pt]). Returns its MathML, or null.
Parser.prototype.optional = function () {
    const tk = this.t[this.i];
    if (!tk || tk.type !== 'other' || tk.v !== '[') return null;
    this.i++;
    const inner = [];
    while (this.t[this.i] && !(this.t[this.i].type === 'other' && this.t[this.i].v === ']')) {
        const before = this.i;
        const a = this.atom();
        if (a !== null) inner.push(this.scripts(a));
        if (this.i === before) this.i++;
    }
    if (this.t[this.i]) this.i++;   // ']'
    return inner.length ? wrap(inner) : '<mrow></mrow>';
};

// Sub/superscripts and primes attach to whatever atom just came out.
Parser.prototype.scripts = function (a) {
    let sub = null, sup = null, primes = '';
    for (;;) {
        const tk = this.t[this.i];
        if (!tk) break;
        if (tk.type === 'sub' && sub === null) { this.i++; sub = this.arg(); continue; }
        if (tk.type === 'sup' && sup === null) { this.i++; sup = this.arg(); continue; }
        if (tk.type === 'other' && tk.v === "'") { this.i++; primes += '′'; continue; }
        if (tk.type === 'cmd' && tk.v === 'prime') { this.i++; primes += '′'; continue; }
        break;
    }
    if (primes) sup = sup ? `<mrow><mo>${primes}</mo>${sup}</mrow>` : `<mo>${primes}</mo>`;
    if (sub === null && sup === null) return a.ml;

    const u = a.under;   // ∑ and lim take limits above/below, ∫ and x beside
    if (sub !== null && sup !== null) {
        return u ? `<munderover>${a.ml}${sub}${sup}</munderover>`
                 : `<msubsup>${a.ml}${sub}${sup}</msubsup>`;
    }
    if (sub !== null) return u ? `<munder>${a.ml}${sub}</munder>` : `<msub>${a.ml}${sub}</msub>`;
    return u ? `<mover>${a.ml}${sup}</mover>` : `<msup>${a.ml}${sup}</msup>`;
};

Parser.prototype.atom = function () {
    const tk = this.t[this.i];
    if (!tk) return null;

    switch (tk.type) {
        case '{': {
            this.i++;
            const inner = this.list(null);
            if (this.t[this.i] && this.t[this.i].type === '}') this.i++;
            return { ml: inner.length ? wrap(inner) : '<mrow></mrow>' };
        }
        case '}':
            return null;
        case 'num':
            this.i++;
            return { ml: `<mn>${esc(tk.v)}</mn>` };
        case 'letter':
            this.i++;
            return { ml: `<mi>${esc(tk.v)}</mi>` };
        case 'other':
            this.i++;
            return { ml: opML(tk.v) };
        case 'amp':
            this.i++;
            return null;
        case 'sub':
        case 'sup':
            return { ml: '<mrow></mrow>' };   // stray script: scripts() takes it
        case 'styled':
            this.i++;
            return { ml: styledML(tk.cmd, tk.v) };
        case 'begin':
            this.i++;
            return { ml: this.env(tk.v) };
        case 'end':
            this.i++;
            return null;
        case 'cmd':
            this.i++;
            return this.command(tk.v);
    }
    return null;
};

Parser.prototype.command = function (name) {
    // Structure
    if (name === 'frac' || name === 'dfrac' || name === 'tfrac' || name === 'cfrac') {
        const a = this.arg(), b = this.arg();
        return { ml: `<mfrac>${a}${b}</mfrac>` };
    }
    if (name === 'binom' || name === 'dbinom' || name === 'tbinom') {
        const a = this.arg(), b = this.arg();
        return { ml: `<mrow><mo stretchy="true">(</mo>`
            + `<mfrac linethickness="0">${a}${b}</mfrac>`
            + `<mo stretchy="true">)</mo></mrow>` };
    }
    if (name === 'sqrt') {
        const index = this.optional();
        const body = this.arg();
        return { ml: index ? `<mroot>${body}${index}</mroot>` : `<msqrt>${body}</msqrt>` };
    }
    if (name === 'left') {
        const open = this.delim();
        const inner = this.list(new Set(['right']));
        let close = '';
        const tk = this.t[this.i];
        if (tk && tk.type === 'cmd' && tk.v === 'right') { this.i++; close = this.delim(); }
        return { ml: `<mrow>${open ? `<mo stretchy="true">${esc(open)}</mo>` : ''}`
            + (inner.length ? wrap(inner) : '')
            + `${close ? `<mo stretchy="true">${esc(close)}</mo>` : ''}</mrow>` };
    }
    if (name === 'right') return null;          // unmatched — swallow
    if (name === 'over') return { ml: '<mo>/</mo>' };
    if (name === 'stackrel' || name === 'overset') {
        const top = this.arg(), base = this.arg();
        return { ml: `<mover>${base}${top}</mover>` };
    }
    if (name === 'underset') {
        const bottom = this.arg(), base = this.arg();
        return { ml: `<munder>${base}${bottom}</munder>` };
    }
    if (name === 'underbrace' || name === 'overbrace') {
        const body = this.arg();
        const brace = name === 'underbrace' ? '⏟' : '⏞';
        const tag = name === 'underbrace' ? 'munder' : 'mover';
        return { ml: `<${tag}>${body}<mo stretchy="true">${brace}</mo></${tag}>`,
                 under: true };
    }
    if (name === 'boxed') return { ml: `<menclose notation="box">${this.arg()}</menclose>` };
    if (name === 'hspace' || name === 'mspace' || name === 'raisebox') { this.arg(); return null; }

    // Accents
    if (ACCENTS[name]) {
        const body = this.arg();
        const wide = name === 'overline' || name === 'bar' || name === 'widehat'
            || name === 'widetilde' || name === 'vec' || name === 'overrightarrow'
            || name === 'overleftarrow';
        return { ml: `<mover accent="true">${body}`
            + `<mo${wide ? ' stretchy="true"' : ''}>${esc(ACCENTS[name])}</mo></mover>` };
    }
    if (name === 'underline') return { ml: `<munder accent="true">${this.arg()}<mo stretchy="true">‾</mo></munder>` };

    // Big operators and named limits
    if (BIG[name]) {
        const [ch, under] = BIG[name];
        return { ml: `<mo${under ? ' movablelimits="true"' : ''}>${esc(ch)}</mo>`, under };
    }
    if (NAMED_LIMITS[name]) {
        return { ml: `<mi>${esc(NAMED_LIMITS[name])}</mi>`, under: true };
    }
    if (FUNCS.has(name)) return { ml: `<mi>${esc(name)}</mi>` };

    // Symbols
    if (GREEK_LOWER[name]) return { ml: `<mi>${esc(GREEK_LOWER[name])}</mi>` };
    if (GREEK_UPPER[name]) return { ml: `<mi mathvariant="normal">${esc(GREEK_UPPER[name])}</mi>` };
    if (OPS[name]) return { ml: `<mo>${esc(OPS[name])}</mo>` };
    if (SPACES[name]) return { ml: `<mspace width="${SPACES[name]}"/>` };

    // Literals: \% \$ \{ \} \_ \# \&
    if (/^[%$#&_{}]$/.test(name)) {
        return { ml: name === '_' ? '<mo>_</mo>' : `<mo>${esc(name)}</mo>` };
    }

    // No-ops: layout hints and numbering that MathML gets from context
    if (name === '\\' || name === 'cr' || name === 'newline') { this.optional(); return null; }
    if (name === 'displaystyle' || name === 'textstyle' || name === 'scriptstyle'
        || name === 'limits' || name === 'nolimits' || name === 'nonumber'
        || name === 'notag' || name === 'hline' || name === 'noalign'
        || name === 'ensuremath' || name === 'phantom') {
        return null;
    }
    if (name === 'tag' || name === 'label' || name === 'ref') { this.arg(); return null; }

    // Unknown: show the name rather than dropping the reader's content.
    return { ml: `<mi>${esc(name)}</mi>` };
};

// The delimiter after \left or \right.
Parser.prototype.delim = function () {
    const tk = this.t[this.i];
    if (!tk) return '';
    if (tk.type === 'other' || tk.type === 'letter' || tk.type === 'num') {
        this.i++;
        return DELIMS[tk.v] !== undefined ? DELIMS[tk.v] : tk.v;
    }
    if (tk.type === '{' || tk.type === '}') { this.i++; return tk.type; }
    if (tk.type === 'cmd') {
        this.i++;
        if (DELIMS[tk.v] !== undefined) return DELIMS[tk.v];
        if (OPS[tk.v]) return OPS[tk.v];
        return '';
    }
    return '';
};

// \begin{env} … \end{env}
Parser.prototype.env = function (rawName) {
    const name = rawName.replace(/\*$/, '');

    // equation/displaymath carry no layout of their own.
    if (name === 'equation' || name === 'displaymath' || name === 'math') {
        const inner = this.list(new Set(['end']));
        if (this.t[this.i] && this.t[this.i].type === 'end') this.i++;
        return inner.length ? wrap(inner) : '<mrow></mrow>';
    }
    // array/alignat take a column spec we don't need — consume and discard.
    if (name === 'array' || name === 'alignat' || name === 'subarray') this.arg();

    const rows = this.rows();
    const cellAlign = name === 'cases' ? ' columnalign="left"'
        : (name === 'aligned' || name === 'align' || name === 'alignat' || name === 'split')
            ? ' columnalign="right left right left right left"' : '';
    const spacing = name === 'cases' ? ' columnspacing="1em"'
        : (name === 'aligned' || name === 'align' || name === 'split') ? ' columnspacing="0"' : '';

    const table = `<mtable${cellAlign}${spacing}>`
        + rows.map(r => '<mtr>' + r.map(c => `<mtd>${c}</mtd>`).join('') + '</mtr>').join('')
        + '</mtable>';

    const fence = ENV_FENCES[name] || ['', ''];
    if (!fence[0] && !fence[1]) return table;
    return '<mrow>'
        + (fence[0] ? `<mo stretchy="true" fence="true">${esc(fence[0])}</mo>` : '')
        + table
        + (fence[1] ? `<mo stretchy="true" fence="true">${esc(fence[1])}</mo>` : '')
        + '</mrow>';
};

Parser.prototype.rows = function () {
    const stop = new Set(['&', 'end', '\\', 'cr', 'newline']);
    const rows = [];
    let row = [];
    for (;;) {
        const cells = this.list(stop);
        row.push(cells.length ? wrap(cells) : '<mrow></mrow>');
        const tk = this.t[this.i];
        if (!tk) break;
        if (tk.type === 'amp') { this.i++; continue; }
        if (tk.type === 'cmd' && (tk.v === '\\' || tk.v === 'cr' || tk.v === 'newline')) {
            this.i++;
            this.optional();             // \\[6pt]
            rows.push(row);
            row = [];
            continue;
        }
        if (tk.type === 'end') { this.i++; break; }
        break;
    }
    // A trailing \\ before \end leaves an empty row; don't render it.
    if (row.some(c => c !== '<mrow></mrow>') || rows.length === 0) rows.push(row);
    return rows;
};

// --- Public entry ---

window.texToMathML = function (tex, display) {
    const body = wrap(new Parser(tokenize(tex)).list(null));
    return `<math xmlns="http://www.w3.org/1998/Math/MathML"`
        + `${display ? ' display="block"' : ''}>${body}</math>`;
};

})();
