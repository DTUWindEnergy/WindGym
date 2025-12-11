# WindGym Documentation

This folder contains the documentation for the WindGym project, using Docusaurus with auto-generated API documentation from Sphinx.

## Structure

```
Docs/
├── docs/                    # Markdown documentation files
│   ├── api/                 # Auto-generated API documentation
│   │   ├── index.md         # API reference overview
│   │   └── modules/         # Module-specific documentation
│   └── ...                  # Other documentation pages
├── sphinx-source/           # Sphinx configuration for API doc generation
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Sphinx index
│   └── modules/             # Module RST files
├── src/                     # Docusaurus source files
├── build-api-docs.sh        # Script to generate API docs from Python docstrings
├── requirements-docs.txt    # Python dependencies for Sphinx
├── docusaurus.config.js     # Docusaurus configuration
├── sidebars.js              # Sidebar configuration
└── package.json             # Node.js dependencies
```

## Building the Documentation

### Prerequisites

1. **Node.js** (v18 or later)
2. **Python** (3.7 or later)
3. **npm** or **yarn**

### Install Dependencies

```bash
cd Docs
npm install

# For API documentation generation
pip install -r requirements-docs.txt
```

### Development Server

Start a local development server:

```bash
npm run start
```

This will open the documentation in your browser at `http://localhost:3000`.

### Building for Production

#### Full Build (with API docs regeneration)

```bash
npm run build:full
```

This will:
1. Run Sphinx to regenerate API documentation from Python docstrings
2. Build the static Docusaurus site

#### Standard Build (without API regeneration)

```bash
npm run build
```

### Regenerating API Documentation

To regenerate only the API documentation from Python docstrings:

```bash
npm run build:api
# or
./build-api-docs.sh
```

This uses Sphinx with the `sphinx-markdown-builder` extension to generate Markdown files from Python docstrings, which are then incorporated into the Docusaurus site.

## Adding Documentation

### Manual Documentation

Add or edit Markdown files in the `docs/` directory. Update `sidebars.js` to include new pages in the navigation.

### API Documentation

The API documentation is auto-generated from Python docstrings using Sphinx. To update:

1. Add/update docstrings in the Python source code
2. Run `npm run build:api` to regenerate the Markdown files
3. The generated files will be placed in `docs/api/`

### Sphinx Configuration

The Sphinx configuration is in `sphinx-source/conf.py`. Key settings:

- **autodoc**: Extracts documentation from docstrings
- **napoleon**: Supports Google/NumPy style docstrings
- **sphinx-markdown-builder**: Outputs Markdown for Docusaurus

## Deployment

The documentation is deployed automatically via GitLab CI/CD to GitLab Pages at:
https://sys.pages.windenergy.dtu.dk/windgym/
