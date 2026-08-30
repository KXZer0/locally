// Markdown -> HTML. Hand-rolled and escape-first.
//
// In here: the renderer and the TeX bridge. THE ORDER OF THE PASSES IS THE
// DESIGN and is documented inline -- anything whose interior must survive the
// inline rules is parked in a \0 placeholder up front and restored last.
// Read the comments before moving a line.
// Not in here: a markdown library. marked.js is not here on purpose: the
// escape-first property, <think> blocks and scheme-restricted links are why
// (see TODONT.md). Streaming lives in markdown/stream-painter.js.

import { escapeHtml } from '../core/format.js';
import { splitThink } from './think-split.js';

// The whole-message render. Still what every non-streaming path uses, and
// what every streaming path falls back to once the last token lands -- so
// the committed DOM is identical to what it has always been.
export function renderMarkdown(text, isStreaming) {
    const split = splitThink(text, isStreaming);
    return split.thinkHtml + renderBody(split.mainText);
}

// TeX -> MathML, via static/js/tex.js. The source arrives HTML-escaped (the
// whole body is escaped up front) and TeX needs the real characters back — &
// separates matrix columns, < and > are relations. The emitter re-escapes
// everything it puts in a text position.
export function renderMath(tex, block) {
    const src = tex
        .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
        .replace(/&amp;/g, '&');
    try {
        return window.texToMathML(src, block);
    } catch (err) {
        // A model can always emit TeX we don't parse. Showing the source beats
        // showing a gap where an equation should be.
        return `<code class="math-raw">${escapeHtml(src)}</code>`;
    }
}
// Markdown -> HTML. Everything is escaped first, so model output can never
// inject markup; only the tags produced below reach the DOM.
//
// The order is the design: anything whose interior must survive the inline
// rules — code, math, backslash escapes, link targets — is pulled out into a
// \0-delimited placeholder up front and restored last. \0 is stripped from the
// input, so a placeholder can never be forged by the model.
export function renderBody(src) {
    let html = escapeHtml(src).replace(/\u0000/g, '');

    const parked = [];
    const park = (markup) => `\u0000P${parked.push(markup) - 1}\u0000`;

    // Fenced code first — the inline rules below would mangle the asterisks
    // and underscores inside it. ~~~ is the alternate fence; both are taken.
    html = html.replace(/(?:```|~~~)([\w+#.-]*)[ \t]*\n?([\s\S]*?)(?:```|~~~)/g,
        (_, lang, code) => {
            const label = lang ? `<span class="code-lang">${lang}</span>` : '';
            return park(`<pre>${label}<code>${code.replace(/\n$/, '')}</code>`
                + `<button class="copy-btn" onclick="copyCode(this)">copy</button></pre>`);
        });
    html = html.replace(/`([^`\n]+)`/g, (_, c) => park(`<code>${c}</code>`));

    // Math. TeX is built from the characters markdown treats as syntax
    // (_ ^ * \), so it has to come out before any of them are interpreted —
    // and after code, so $x$ inside a code span stays literal.
    const math = (tex, block) => park(renderMath(tex, block));
    html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => math(tex, true));
    html = html.replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => math(tex, true));
    // A bare environment with no delimiters around it — models emit these.
    html = html.replace(
        /\\begin\{(equation\*?|align(?:ed|at)?\*?|gather(?:ed)?\*?|multline\*?|split|cases|array|[pbBvV]?matrix)\}[\s\S]*?\\end\{\1\}/g,
        (m) => math(m, true));
    html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => math(tex, false));
    // Single $ is the ambiguous one: it also writes money. Requiring no space
    // just inside the delimiters and no digit right after the closing $ leaves
    // "$5 and $10" alone while still catching $x^2$. The closing-side rule is
    // the final [^\s$\\] rather than a lookbehind, which older Safari can't
    // parse — and one unparsable regex takes the whole file down with it.
    html = html.replace(/(^|[^\\$\w])\$(?!\s)((?:[^$\n\\]|\\.){0,299}?[^\s$\\])\$(?!\d)/g,
        (_, pre, tex) => pre + math(tex, false));

    // Backslash escapes, parked so the inline rules never see the character.
    html = html.replace(/\\([\\`*_{}\[\]()#+\-.!$|~])/g, (_, c) => park(escapeHtml(c)));

    // Tables: | a | b |  /  |---|---|  /  rows.  A :---: separator sets the
    // column's alignment, which is the only thing GFM puts in that row.
    html = html.replace(
        // The separator row is |---|---|, so the character class has to allow
        // the inner pipes as well as the dashes.
        /(^\|.+\|[ \t]*\n\|[ \t:|-]+\|?[ \t]*\n(?:\|.*\|[ \t]*(?:\n|$))+)/gm,
        (block) => {
            const rows = block.trim().split('\n');
            const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
            const head = cells(rows[0]);
            const align = cells(rows[1]).map(s =>
                /^:-+:$/.test(s) ? ' style="text-align:center"'
                    : /^-+:$/.test(s) ? ' style="text-align:right"' : '');
            const body = rows.slice(2).map(cells);
            const at = (i) => align[i] || '';
            return '<table><thead><tr>'
                + head.map((c, i) => `<th${at(i)}>${c}</th>`).join('')
                + '</tr></thead><tbody>'
                + body.map(r => '<tr>' + r.map((c, i) => `<td${at(i)}>${c}</td>`).join('') + '</tr>').join('')
                + '</tbody></table>\n';
        });

    // Block constructs, line by line. Lists carry an indent stack so nesting
    // is real nesting: the parent <li> stays open around the child list.
    const out = [];
    const lists = [];                  // { tag, indent, itemOpen }
    let inQuote = false, pendingBlank = false;

    const closeTo = (indent) => {
        while (lists.length && lists[lists.length - 1].indent > indent) {
            const top = lists.pop();
            if (top.itemOpen) out.push('</li>');
            out.push(`</${top.tag}>`);
        }
    };
    const closeLists = () => closeTo(-1);
    const closeQuote = () => { if (inQuote) { out.push('</blockquote>'); inQuote = false; } };
    // A blank line inside a list is held rather than acted on: "- a\n\n- b" is
    // one loose list, not two lists.
    const flushBlank = () => {
        if (!pendingBlank) return;
        pendingBlank = false;
        closeLists(); closeQuote(); out.push('');
    };

    for (const raw of html.split('\n')) {
        const line = raw.trimEnd();

        if (/^\s*$/.test(line)) {
            if (lists.length) { pendingBlank = true; continue; }
            closeQuote(); out.push(''); continue;
        }

        if (/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line)) {
            pendingBlank = false;
            closeLists(); closeQuote(); out.push('<hr>'); continue;
        }

        const indent = raw.match(/^[ \t]*/)[0].replace(/\t/g, '    ').length;
        const ul = line.match(/^[ \t]*[-*+][ \t]+(.*)$/);
        const ol = line.match(/^[ \t]*(\d+)[.)][ \t]+(.*)$/);

        if (ul || ol) {
            pendingBlank = false;
            closeQuote();
            const tag = ul ? 'ul' : 'ol';
            let content = ul ? ul[1] : ol[2];
            let attrs = '';
            const task = content.match(/^\[([ xX])\][ \t]+(.*)$/);   // GFM checkbox
            if (task) {
                attrs = ' class="task"';
                content = `<input type="checkbox" disabled`
                    + `${task[1] === ' ' ? '' : ' checked'}> ${task[2]}`;
            }

            closeTo(indent);
            let top = lists[lists.length - 1];
            if (!top || top.indent < indent) {
                out.push(`<${tag}>`);                    // nested inside the open <li>
                lists.push({ tag, indent, itemOpen: false });
            } else if (top.tag !== tag) {
                if (top.itemOpen) out.push('</li>');
                out.push(`</${top.tag}>`);
                lists.pop();
                out.push(`<${tag}>`);
                lists.push({ tag, indent, itemOpen: false });
            }
            top = lists[lists.length - 1];
            if (top.itemOpen) out.push('</li>');
            out.push(`<li${attrs}>${content}`);
            top.itemOpen = true;
            continue;
        }

        // An indented line under an open item is that item's continuation.
        const open = lists[lists.length - 1];
        if (open && open.itemOpen && indent >= open.indent + 2) {
            pendingBlank = false;
            out.push(line.trim());
            continue;
        }
        flushBlank();

        const h = line.match(/^(#{1,6})\s+(.*?)\s*#*$/);
        if (h) {
            closeLists(); closeQuote();
            const lvl = Math.min(6, h[1].length + 1);   // #  ->  h2, page owns h1
            out.push(`<h${lvl}>${h[2]}</h${lvl}>`);
            continue;
        }

        // Setext heading, '=' underline only: a '-' underline is far more often
        // meant as a rule than as a heading in chat output.
        const prev = out[out.length - 1];
        if (/^\s*=+\s*$/.test(line) && prev && /\S/.test(prev)
            && !/^\s*<(?:h[1-6]|ul|ol|li|blockquote|table|hr|pre)/.test(prev)) {
            out[out.length - 1] = `<h2>${prev.trim()}</h2>`;
            continue;
        }

        const q = line.match(/^\s*&gt;\s?(.*)$/);       // '>' is escaped by now
        if (q) {
            closeLists();
            if (!inQuote) { out.push('<blockquote>'); inQuote = true; }
            out.push(q[1] || '<br>');
            continue;
        }
        closeQuote();

        closeLists();
        out.push(line);
    }
    flushBlank();
    closeLists(); closeQuote();
    html = out.join('\n');

    // Images, then links. Both targets are rebuilt from an escaped string and
    // restricted by scheme, so a model can't emit javascript: and have it
    // survive. Only the tags are parked — the link text stays in the stream so
    // the emphasis rules below still reach it.
    html = html.replace(/!\[([^\]]*)\]\(((?:https?:\/\/|data:image\/)[^\s)]+)\)/g,
        (_, alt, src2) => park(`<img src="${src2}" alt="${alt}" loading="lazy">`));
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        (_, txt, href) =>
            park(`<a href="${href}" target="_blank" rel="noopener noreferrer">`)
            + txt + park('</a>'));
    // Autolinks. Explicit links are parked by now, so this can't re-link an
    // href it already produced.
    const autolink = (u) =>
        park(`<a href="${u}" target="_blank" rel="noopener noreferrer">`) + u + park('</a>');
    html = html.replace(/&lt;(https?:\/\/[^\s&]+)&gt;/g, (_, u) => autolink(u));
    html = html.replace(/(^|[\s(])(https?:\/\/[^\s<)\]]*[^\s<>)\].,;:!?'"])/g,
        (_, pre, u) => pre + autolink(u));

    // Inline spans
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/___(.+?)___/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    html = html.replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*/g, '$1<em>$2</em>');
    // Underscore emphasis only at word boundaries, so snake_case survives.
    html = html.replace(/(^|[\s([{])_(?!\s)([^_\n]+?)_(?=$|[\s.,;:!?)\]}])/g, '$1<em>$2</em>');
    html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');

    // Paragraph breaks, then single newlines — but never inside block tags,
    // where a <br> would add stray gaps.
    html = html
        .split(/\n{2,}/)
        .map(chunk => /^\s*<(?:h[1-6]|ul|ol|pre|blockquote|table|hr)/.test(chunk.trim())
            ? chunk
            : (chunk.trim() ? `<p>${chunk.trim()}</p>` : ''))
        .join('\n');
    html = html.replace(/(<p>[\s\S]*?<\/p>)/g, m => m.replace(/\n/g, '<br>'));

    // Restore everything that was parked, in one pass — parked markup can't
    // contain a placeholder, since \0 was stripped from the input.
    return html.replace(/\u0000P(\d+)\u0000/g, (_, i) => parked[i]);
}
export function copyCode(btn) {
    const code = btn.parentElement.querySelector('code').textContent;
    navigator.clipboard.writeText(code);
    btn.textContent = 'copied';
    setTimeout(() => btn.textContent = 'copy', 1500);
}
window.copyCode = copyCode;
