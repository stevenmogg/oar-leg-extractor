## Codespaces / Dev Container

This repository includes a ready-to-use GitHub Codespaces configuration.

### How to use
1. Open the repository on GitHub.
2. Click the green "Code" button → "Create codespace on main".
3. Wait for the container to build (a few minutes on first launch).
4. Once ready, run the task:
   - Press Cmd/Ctrl+Shift+B and choose "Run Streamlit"; or
   - Open a terminal and run:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The port 8501 is forwarded automatically and will open in your browser.

### Notes
- Dependencies are installed automatically via `postCreateCommand`.
- If you add new Python packages, run `pip freeze > requirements.txt` to update dependencies for future Codespaces sessions.