/**
 * DereInside Obsidian Plugin
 *
 * Features:
 * - Search derekinside knowledge base
 * - Embed search results as notes
 * - Mine current vault to derekinside
 *
 * Configuration:
 * - DereInside URL (default: http://localhost:18890)
 * - Auth token (optional)
 */

const { Plugin, Modal, Setting, Notice, Platform } = require('obsidian');

const DEFAULT_SETTINGS = {
  dereInsideUrl: 'http://localhost:18890',
  authToken: '',
  maxResults: 20,
  autoMine: false,
};

// ── Settings Tab ──────────────────────────────────────────────

class DereInsideSettingTab {
  constructor(app, plugin) {
    this.app = app;
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName('DereInside URL')
      .setDesc('The URL of your DereInside HTTP bridge')
      .addText(text => text
        .setPlaceholder('http://localhost:18890')
        .setValue(this.plugin.settings.dereInsideUrl)
        .onChange(async val => {
          this.plugin.settings.dereInsideUrl = val.replace(/\/+$/, '');
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName('Auth Token')
      .setDesc('Optional: X-DEREINSIDE-TOKEN (leave blank if auth disabled)')
      .addText(text => text
        .setPlaceholder('')
        .setValue(this.plugin.settings.authToken)
        .onChange(async val => {
          this.plugin.settings.authToken = val;
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName('Max Results')
      .setDesc('Number of search results to display')
      .addSlider(slider => slider
        .setLimits(5, 50, 5)
        .setValue(this.plugin.settings.maxResults)
        .setDynamicTooltip()
        .onChange(async val => {
          this.plugin.settings.maxResults = val;
          await this.plugin.saveSettings();
        }));
  }
}

// ── Search Modal ──────────────────────────────────────────────

class DereInsideSearchModal extends Modal {
  constructor(app, plugin) {
    super(app);
    this.plugin = plugin;
    this.query = '';
    this.results = [];
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.addClass('dere-modal');

    // Input
    this.inputEl = contentEl.createEl('input', {
      type: 'text',
      cls: 'dere-search-input',
      placeholder: 'Search DereInside... (Enter to search)',
    });
    this.inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.doSearch();
    });
    setTimeout(() => this.inputEl.focus(), 100);

    // Results container
    this.resultsEl = contentEl.createDiv({ cls: 'dere-results' });
    this.resultsEl.createDiv({ cls: 'dere-empty', text: 'Type a query and press Enter' });

    // Status bar
    this.statusEl = contentEl.createDiv({ cls: 'dere-status-bar' });
    this.updateStatus('Ready');
  }

  async doSearch() {
    const q = this.inputEl.value.trim();
    if (!q) return;

    this.resultsEl.empty();
    this.resultsEl.createDiv({ cls: 'dere-empty', text: 'Searching...' });
    this.updateStatus('Searching...');

    const url = `${this.plugin.settings.dereInsideUrl}/api/v1/search`;
    const headers = { 'Content-Type': 'application/json' };
    if (this.plugin.settings.authToken) {
      headers['X-DEREINSIDE-TOKEN'] = this.plugin.settings.authToken;
    }

    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({ query: q, top_k: this.plugin.settings.maxResults }),
        signal: AbortSignal.timeout(60000),
      });

      if (!resp.ok) {
        this.showError(`HTTP ${resp.status}: ${resp.statusText}`);
        return;
      }

      const data = await resp.json();
      this.results = data.results || [];
      this.renderResults(q, data);
    } catch (e) {
      this.showError(`Connection failed: ${e.message}`);
    }
  }

  renderResults(query, data) {
    this.resultsEl.empty();

    if (this.results.length === 0) {
      this.resultsEl.createDiv({ cls: 'dere-empty', text: 'No results found.' });
      this.updateStatus('0 results');
      return;
    }

    this.updateStatus(`${data.total} results (${data.timing_ms}ms${data.cache_hit ? ', cached' : ''})`);

    this.results.forEach(r => {
      const card = this.resultsEl.createDiv({ cls: 'dere-result' });
      card.addEventListener('click', () => this.insertResult(r));

      const meta = card.createDiv({ cls: 'dere-meta' });
      if (r.wing || r.room) meta.createEl('span', { text: `🏛️ ${r.wing || ''}/${r.room || ''}` });
      if (r.source_path) meta.createEl('span', { text: `📄 ${r.source_path.split('/').pop()}` });
      meta.createEl('span', { text: `score: ${(r.score || 0).toFixed(3)}` });

      const preview = card.createDiv({ cls: 'dere-preview' });
      preview.innerHTML = this.highlight(r.chunk_text, query);
    });
  }

  highlight(text, query) {
    if (!query) return text;
    const words = query.split(/\s+/).filter(w => w.length > 1);
    let result = text.substring(0, 500);
    words.forEach(w => {
      const re = new RegExp(`(${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      result = result.replace(re, '<em>$1</em>');
    });
    return result.replace(/\n/g, '<br>');
  }

  insertResult(result) {
    const editor = this.app.workspace.getMostRecentLeaf()?.view?.editor;
    if (!editor) {
      new Notice('No active editor');
      return;
    }

    const line = editor.getCursor().line;
    const prefix = result.source_path ? `> **Source:** \`${result.source_path}\`` : '';
    const score = (result.score || 0).toFixed(3);

    const block = [
      `> 🧠 **DereInside** — ${result.wing || ''}/${result.room || ''} (score: ${score})`,
      prefix,
      `> ${result.chunk_text.substring(0, 300).replace(/\n/g, '\n> ')}`,
      '',
    ].filter(l => l).join('\n');

    editor.replaceRange(block, { line, ch: 0 });
    new Notice('DereInside result inserted');
    this.close();
  }

  showError(msg) {
    this.resultsEl.empty();
    this.resultsEl.createDiv({ cls: 'dere-error', text: msg });
    this.updateStatus('Error');
  }

  updateStatus(text) {
    if (this.statusEl) this.statusEl.setText(text);
  }

  onClose() {
    const { contentEl } = this;
    contentEl.empty();
  }
}

// ── Mine Modal ────────────────────────────────────────────────

class DereInsideMineModal extends Modal {
  constructor(app, plugin) {
    super(app);
    this.plugin = plugin;
    this.status = '';
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.addClass('dere-modal');
    contentEl.createEl('h3', { text: '📦 Mine Vault to DereInside', attr: { style: 'padding:16px;margin:0;border-bottom:1px solid var(--dere-border)' } });
    this.statusEl = contentEl.createDiv({ cls: 'dere-empty', text: 'Preparing to mine vault...' });
    this.doMine();
  }

  async doMine() {
    this.statusEl.setText('Mining vault...');

    const vault = this.app.vault;
    const files = vault.getMarkdownFiles();
    const total = files.length;
    let mined = 0, failed = 0;

    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      this.statusEl.setText(`[${i+1}/${total}] ${f.path}`);

      try {
        const content = await vault.read(f);
        const resp = await fetch(
          `${this.plugin.settings.dereInsideUrl}/api/v1/mine`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              path: f.path,
              wing: 'obsidian',
              room: f.parent?.path || 'root',
              content: content,
            }),
          }
        );
        if (resp.ok) mined++;
        else failed++;
      } catch (e) {
        failed++;
      }
    }

    this.statusEl.setText(`✅ Done: ${mined} files mined, ${failed} failed`);
    new Notice(`DereInside: ${mined}/${total} files mined`);
    setTimeout(() => this.close(), 2000);
  }

  onClose() {
    const { contentEl } = this;
    contentEl.empty();
  }
}

// ── Plugin ────────────────────────────────────────────────────

module.exports = class DereInsidePlugin extends Plugin {
  async onload() {
    await this.loadSettings();

    // Search command
    this.addCommand({
      id: 'derekinside-search',
      name: 'Search DereInside',
      callback: () => {
        new DereInsideSearchModal(this.app, this).open();
      },
    });

    // Mine vault command
    this.addCommand({
      id: 'derekinside-mine-vault',
      name: 'Mine vault to DereInside',
      callback: () => {
        new DereInsideMineModal(this.app, this).open();
      },
    });

    // Ribbon icon
    this.addRibbonIcon('search', 'DereInside Search', () => {
      new DereInsideSearchModal(this.app, this).open();
    });

    // Settings tab
    this.addSettingTab(new DereInsideSettingTab(this.app, this));

    // Auto-mine on vault open (if enabled)
    if (this.settings.autoMine) {
      this.app.workspace.onLayoutReady(() => {
        new Notice('DereInside: Auto-mining vault...');
        new DereInsideMineModal(this.app, this).open();
      });
    }

    console.log('DereInside plugin loaded');
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }
};
