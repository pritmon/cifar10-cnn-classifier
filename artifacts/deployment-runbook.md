# 🚀 Deployment Runbook — CIFAR-10 CNN Classifier

A step-by-step operational guide for deploying, updating, and troubleshooting
the live web app. Written so it can be followed cold, months later.

| | |
|---|---|
| **Live app** | https://huggingface.co/spaces/prithugging/cifar10-cnn-classifier |
| **Source repo** | https://github.com/pritmon/cifar10-cnn-classifier |
| **Platform** | Hugging Face Spaces (free CPU tier) |
| **Framework** | Gradio 5 |
| **Model served** | `cifar10_model_augmented.keras` (~75.4% test accuracy) |

---

## 1. What gets deployed

The deploy bundle lives in [`huggingface_space/`](../huggingface_space) and is
**self-contained** — every file the Space needs sits in one flat folder:

| File | Purpose |
|------|---------|
| `app.py` | The Gradio web app (flat paths — model & images in the same folder) |
| `requirements.txt` | Python dependencies (Gradio is provided by the Space SDK) |
| `README.md` | Space config (YAML frontmatter: SDK, version, app file) |
| `cifar10_model_augmented.keras` | The trained model |
| `cat.png`, `dog.png`, `ship.png`, `airplane.png`, `frog.png` | Example images |

---

## 2. Prerequisites

- A free Hugging Face account: https://huggingface.co/join
- The files in `huggingface_space/` (already in this repo)

---

## 3. First-time deploy

1. **Create the Space** → https://huggingface.co/new-space
   - Space name: `cifar10-cnn-classifier`
   - License: `mit`
   - SDK: **Gradio**
   - Hardware: **CPU basic** (free)
   - Visibility: **Public**
   - Click **Create Space**.
2. **Upload the bundle** → on the Space, go to **Files → + Add file → Upload files**.
   - Drag in **all** files from `huggingface_space/`.
   - Overwrite the auto-generated `README.md` when prompted (ours has the correct config).
   - Click **Commit changes to main**.
3. **Wait for the build** (~3–5 min): status goes 🔵 *Building* → 🟢 *Running*.
4. **Verify** (see §6).

---

## 4. Updating an existing deploy

Only re-upload the file(s) that changed:

| You changed… | Re-upload… |
|--------------|-----------|
| The model (retrained) | `cifar10_model_augmented.keras` |
| The UI / prediction code | `app.py` |
| Dependencies | `requirements.txt` |
| Space config (SDK version, title) | `README.md` |

Files → Add file → Upload files → drag the changed file → overwrite → commit.
The Space rebuilds automatically.

> Keep the repo copy in sync: update the matching file under `huggingface_space/`
> and commit to GitHub too, so the bundle stays the source of truth.

---

## 5. Configuration reference

**`huggingface_space/README.md` frontmatter** (controls the Space):

```yaml
sdk: gradio
sdk_version: 5.49.1     # Gradio 5 — required for Python 3.13 compatibility
app_file: app.py
```

**`huggingface_space/requirements.txt`** (only model deps — Gradio comes from the SDK):

```
tensorflow-cpu
pillow
numpy
audioop-lts; python_version >= "3.13"
```

---

## 6. Verifying a deploy

1. **Status badge** on the Space reads 🟢 **Running**.
2. **Smoke test in the UI:** open the **App** tab, click the `cat` example → it should
   return `cat` as the top guess; click `ship` → returns `ship`.
3. **Reachability check** (from a terminal):
   ```bash
   curl -s -o /dev/null -w "%{http_code}" \
     https://huggingface.co/spaces/prithugging/cifar10-cnn-classifier
   # expect: 200
   ```

---

## 7. Troubleshooting

Read logs on the Space: **App tab → Logs**. Two log views matter:
- **Build** — dependency install (pip). Failures here = bad `requirements.txt`.
- **Container** — the running app's output. Startup crashes show here.

### Known issues & fixes (already resolved in this repo)

| Symptom (in Container logs) | Cause | Fix |
|-----------------------------|-------|-----|
| `ModuleNotFoundError: No module named 'audioop'` | Python 3.13 removed the stdlib `audioop` module that Gradio's `pydub` imports | `audioop-lts; python_version >= "3.13"` in `requirements.txt` |
| `TypeError: unhashable type: 'dict'` during page render | Gradio 4.x clashing with newer web libraries on Python 3.13 | Use **Gradio 5** (`sdk_version: 5.49.1`) |
| `When localhost is not accessible, a shareable link must be created` | A downstream effect of the render crash above | Fixed by the Gradio 5 upgrade |

### General tips
- A green **Building** that never turns Running → check **Build** logs for a pip error.
- App shows **Runtime error** → check **Container** logs; scroll to the bottom for the
  real exception (ignore the `anyio`/`uvicorn` framework frames).
- Always confirm the model filename in `app.py` (`MODEL_PATH`) matches the uploaded file.

---

## 8. Rollback

Every upload is a commit on the Space's git history.
**Settings → (or the Files commit history) → revert** to a previous working commit,
or simply re-upload the last-known-good `app.py` / `requirements.txt` / `README.md`.

---

## 9. Local dry-run (optional, before deploying)

Run the app on your own machine first:

```bash
source venv/bin/activate
python app.py            # open the printed http://127.0.0.1:7860
```

If it works locally, the Space build is very likely to succeed too.
