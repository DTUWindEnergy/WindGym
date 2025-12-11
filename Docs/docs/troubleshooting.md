# Troubleshooting & FAQ

This page covers common issues and frequently asked questions about WindGym.

---

## Installation Issues

### Pixi Installation Fails

**Problem**: `pixi install` fails with dependency resolution errors

**Solutions**:

1. **Clear the cache and retry:**

   ```bash
   pixi clean cache-dir
   pixi install
   ```

2. **Check your pixi version:**

   ```bash
   pixi --version
   # Should be >= 0.10.0
   ```

3. **Update pixi:**

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

4. **Try installing with verbose output:**
   ```bash
   pixi install -vv
   ```

---

### Import Errors After Installation

**Problem**: `ModuleNotFoundError: No module named 'WindGym'`

**Solutions**:

1. **Ensure the environment is activated:**

   ```bash
   pixi shell
   python -c "import WindGym; print('Success!')"
   ```

2. **Check if WindGym is installed in editable mode:**

   ```bash
   pip list | grep WindGym
   # Should show: WindGym 0.0.2 /path/to/windgym
   ```

3. **Reinstall in editable mode:**
   ```bash
   pixi run install
   # or manually:
   pip install -e .
   ```

---

## Related Pages

- [Installation Guide](installation.md)
- [Core Concepts](concepts.md)
- [API Reference](api/index.md)
- [Developer Guidelines](developer-guidelines.md)
