# Electron Build Guide for Hwaseung Dashboard

## Prerequisites

- Node.js 16 or higher
- Python 3.10 (for development)
- Windows (this guide is for Windows build)

## Build Steps

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Setup Python Runtime (for production build)

Run the PowerShell script to download and configure Python embedded runtime:

```powershell
cd frontend
.\setup-python-runtime.ps1
```

This will:
- Download Python 3.10 Embedded
- Install pip
- Install all packages from `backend/requirements.txt`
- Configure the runtime in `python-runtime/` folder

### 3. Development Mode

For development, you can use your system Python:

**Terminal 1: Start React Dev Server**
```bash
cd frontend
npm start
```

**Terminal 2: Start Backend**
```bash
cd backend
python run.py
```

**Terminal 3: Start Electron**
```bash
cd frontend
npm run electron:dev
```

Or use the combined development script:
```bash
cd frontend
npm run electron:dev
# Make sure React dev server (port 3000) and backend (port 8000) are already running
```

### 4. Production Build

```bash
cd frontend

# Build React app
npm run build

# Build Electron app (includes TypeScript compilation and electron-builder)
npm run electron:build
```

The output will be in `frontend/dist/`:
- `Hwaseung-Dashboard-X.X.X-portable.exe` - Portable executable (no installation required)

## File Structure

```
frontend/
├── electron/              # Electron main process
│   ├── main.ts           # Main process logic
│   ├── preload.ts        # Preload script
│   └── tsconfig.json     # TypeScript config for Electron
├── python-runtime/       # Embedded Python (created by setup script)
│   ├── python.exe
│   └── ... (Python files)
├── build/                # React production build
├── dist/                 # Electron build output
└── package.json          # Contains Electron config

backend/
├── app/                  # FastAPI application
├── run.py               # Backend entry point (Electron-aware)
└── requirements.txt     # Python dependencies
```

## Key Features

### Electron Configuration (package.json)

- **main**: Points to `electron/main.js` (compiled from main.ts)
- **homepage**: Set to `./` for relative paths
- **build**: electron-builder configuration
  - Bundles Python runtime, backend code, and database
  - Creates portable .exe (no installation required)
  - Uses Windows x64 target

### Backend Modifications (backend/run.py)

- Detects Electron mode via `ELECTRON_MODE` environment variable
- Forces UTF-8 encoding for Windows compatibility
- Uses `localhost` instead of `0.0.0.0` in Electron mode
- Disables hot-reload in production

### Main Process (electron/main.ts)

- Manages Python backend subprocess
- Runs Express server for frontend (production mode)
- Handles database initialization (copies to user data folder)
- Provides logging to user data directory
- Supports both development and production modes

## Scripts

| Script | Description |
|--------|-------------|
| `npm start` | Start React dev server (port 3000) |
| `npm run build` | Build React app for production |
| `npm run electron` | Run Electron (production mode) |
| `npm run electron:dev` | Run Electron (development mode with DevTools) |
| `npm run electron:build` | Build complete Electron app with all resources |
| `npm run postinstall` | Install Electron app dependencies |

## Troubleshooting

### Python Runtime Issues

If packages fail to install:
```powershell
cd frontend/python-runtime
.\python.exe -m pip install <package-name>
```

### Port Conflicts

- Frontend: Port 3000 (dev), Port 3000 (prod via Express)
- Backend: Port 8000

Make sure these ports are available.

### Build Errors

If `electron-builder` fails:
1. Check that `python-runtime/` exists and has all dependencies
2. Verify `build/` folder exists (run `npm run build` first)
3. Check that database file exists: `../hwaseung_RnD.db`

### Logs

In production, logs are saved to:
- Windows: `%APPDATA%/Hwaseung Dashboard/logs/app.log`

## Distribution

The portable .exe can be distributed as-is:
- No installation required
- Self-contained (includes Python, backend, frontend, DB)
- First run creates user data folder with database copy
- Approximately 300-500 MB (depending on dependencies)

## Notes

- The database is copied to user data folder on first run
- Each user gets their own database copy
- Logs are saved per-user in AppData
- Python subprocess is automatically managed (started/stopped with app)
