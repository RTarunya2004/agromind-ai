# AgroMind AI — Research Prototype Website

## What is included
- Static farmer-facing frontend (`frontend/`).
- FastAPI prediction backend (`backend/`) with the included reconstructed research checkpoint and preprocessing artifacts.
- Render deployment manifest (`render.yaml`) and Vercel static-site config (`vercel.json`).

## Run locally (Windows)
1. Install Python 3.10 or 3.11 (recommended for PyTorch compatibility).
2. Open a terminal in this project folder.
3. Run:
   ```powershell
   cd backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
4. In a second terminal, serve the frontend folder (avoid opening index.html as a file):
   ```powershell
   cd frontend
   python -m http.server 5500
   ```
5. Open `http://127.0.0.1:5500`. The frontend defaults to same-origin API calls, so for local development set the backend origin in `frontend/api-config.js` to `http://127.0.0.1:8000`:
   ```js
   window.AGROMIND_API_BASE = "http://127.0.0.1:8000";
   ```

## Deploy backend on Render
1. Push this project to GitHub.
2. In Render, create a new Blueprint from the repository and select the included `render.yaml`, or create a Python web service manually.
3. For a manual service, set Root Directory to `backend`, Build Command to `pip install -r requirements.txt`, and Start Command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. After deploy, open `https://YOUR-BACKEND.onrender.com/api/health`; it should return JSON.

## Deploy frontend on Vercel
1. Import the repository in Vercel. Set the Root Directory to `frontend` if deploying the frontend folder as its own project.
2. Before deployment, edit `frontend/api-config.js` and replace the empty value with your deployed backend origin, e.g. `https://YOUR-BACKEND.onrender.com` (no trailing slash). Commit and redeploy.
3. The API currently permits cross-origin requests. For production, restrict CORS in `backend/main.py` to your exact Vercel domain.

## Model and safety limitations
The backend uses `best_reconstructed_hybrid.pth` with a reconstructed TabNet-inspired mask module + Transformer classifier; this is not an exact original TabNet checkpoint. The farmer form supplies seven numeric and three categorical fields. Several training columns are absent and zero-filled, with warnings returned in the API response. Model scores are not calibrated probabilities of farm success. The prediction output is experimental decision support, not validated agronomic advice.

The API health endpoint and Python syntax can be checked locally, but this package has not been represented as field-validated. Do not advertise the paper-reported accuracy as live farm accuracy. Run the backend and test `/api/health` and `/api/predict` in the deployment environment before public use.
