# DereInside Obsidian Plugin 🧠

Search your derekinside knowledge base from Obsidian.
Mine your Obsidian vault into derekinside.

## Installation

1. Make sure DereInside HTTP bridge is running: `derekinside serve --mode http`
2. Copy the `integrations/obsidian-plugin/` folder to your vault's `.obsidian/plugins/derekinside/`
   ```bash
   cp -r ~/derekinside/integrations/obsidian-plugin ~/my-vault/.obsidian/plugins/derekinside
   ```
3. Restart Obsidian (or reload community plugins)
4. Enable **DereInside** in Settings → Community Plugins
5. Configure: Settings → DereInside → set URL (default: `http://localhost:18890`)

## Usage

### Search (`Ctrl+P → "Search DereInside"`)
- Opens a search modal
- Type query, press Enter
- Click a result to insert it into the current note as a blockquote

### Mine Vault (`Ctrl+P → "Mine vault to DereInside"`)
- Ingests all markdown files into derekinside under the `obsidian` wing
- Each file becomes a page with chunked+embedded content

### Ribbon Icon
Click the search icon in the left ribbon to quickly open the search modal.

## Development

The plugin is plain JavaScript (no build step needed).
- `manifest.json` — plugin metadata
- `main.js` — plugin code
- `styles.css` — styling
