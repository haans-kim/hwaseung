"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
const express_1 = __importDefault(require("express"));
const electron_is_dev_1 = __importDefault(require("electron-is-dev"));
const BACKEND_PORT = 8000;
const FRONTEND_PORT = 3000;
const APP_NAME = 'Hwaseung Dashboard';
const DB_NAME = 'hwaseung_RnD.db';
let mainWindow = null;
let backendProcess = null;
let expressServer = null;
// 로그 디렉토리 및 파일 설정
const userDataPath = electron_1.app.getPath('userData');
const logDir = path.join(userDataPath, 'logs');
const logFile = path.join(logDir, 'app.log');
// 로그 디렉토리 생성
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
}
// 로깅 함수
function log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level}] ${message}\n`;
    console.log(logMessage.trim());
    fs.appendFileSync(logFile, logMessage);
}
// DB 파일 초기화
function initializeDatabase() {
    log('Initializing database...');
    const userDbPath = path.join(userDataPath, DB_NAME);
    // 이미 DB가 있으면 사용
    if (fs.existsSync(userDbPath)) {
        log(`Database already exists at: ${userDbPath}`);
        return userDbPath;
    }
    // 리소스에서 DB 복사
    let sourceDbPath;
    if (electron_is_dev_1.default) {
        // 개발 모드: 프로젝트 루트에서 DB 찾기
        sourceDbPath = path.join(__dirname, '..', '..', DB_NAME);
    }
    else {
        // 프로덕션 모드: resources 폴더에서 DB 찾기
        sourceDbPath = path.join(process.resourcesPath, DB_NAME);
    }
    if (fs.existsSync(sourceDbPath)) {
        log(`Copying database from: ${sourceDbPath}`);
        fs.copyFileSync(sourceDbPath, userDbPath);
        log(`Database copied to: ${userDbPath}`);
    }
    else {
        log(`WARNING: Database not found at: ${sourceDbPath}`, 'ERROR');
    }
    return userDbPath;
}
// Python 백엔드 시작
function startBackend() {
    return new Promise((resolve, reject) => {
        log('Starting Python backend...');
        let pythonPath;
        let backendPath;
        if (electron_is_dev_1.default) {
            // 개발 모드: 시스템 Python 사용
            pythonPath = 'python';
            backendPath = path.join(__dirname, '..', '..', 'backend');
        }
        else {
            // 프로덕션 모드: 번들된 Python 사용
            const pythonRuntimePath = path.join(process.resourcesPath, 'python-runtime');
            pythonPath = path.join(pythonRuntimePath, 'python.exe');
            backendPath = path.join(process.resourcesPath, 'backend');
        }
        log(`Python path: ${pythonPath}`);
        log(`Backend path: ${backendPath}`);
        // 환경 변수 설정
        const env = {
            ...process.env,
            ELECTRON_MODE: '1',
            PORT: BACKEND_PORT.toString(),
            PYTHONIOENCODING: 'utf-8',
        };
        const runScript = path.join(backendPath, 'run.py');
        log(`Executing: ${pythonPath} ${runScript}`);
        // Python 프로세스 시작
        backendProcess = (0, child_process_1.spawn)(pythonPath, [runScript], {
            cwd: backendPath,
            env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        // stdout 로깅
        backendProcess.stdout?.on('data', (data) => {
            const message = data.toString().trim();
            if (message) {
                log(`[Backend] ${message}`);
            }
        });
        // stderr 로깅
        backendProcess.stderr?.on('data', (data) => {
            const message = data.toString().trim();
            if (message) {
                log(`[Backend Error] ${message}`, 'ERROR');
            }
        });
        // 프로세스 종료 처리
        backendProcess.on('close', (code) => {
            log(`Backend process exited with code ${code}`);
            backendProcess = null;
        });
        backendProcess.on('error', (error) => {
            log(`Failed to start backend: ${error.message}`, 'ERROR');
            reject(error);
        });
        // 백엔드 시작 대기 (5초)
        setTimeout(() => {
            log('Backend should be ready');
            resolve();
        }, 5000);
    });
}
// Express 프론트엔드 서버 시작
function startFrontend() {
    return new Promise((resolve, reject) => {
        log('Starting Express frontend server...');
        const expressApp = (0, express_1.default)();
        let buildPath;
        if (electron_is_dev_1.default) {
            // 개발 모드에서는 React dev server 사용 (포트 3000)
            log('Development mode: Using React dev server');
            resolve();
            return;
        }
        else {
            // 프로덕션 모드: 빌드된 파일 서빙
            buildPath = path.join(__dirname, '..', 'build');
        }
        log(`Serving static files from: ${buildPath}`);
        expressApp.use(express_1.default.static(buildPath));
        expressApp.get('*', (req, res) => {
            res.sendFile(path.join(buildPath, 'index.html'));
        });
        expressServer = expressApp.listen(FRONTEND_PORT, () => {
            log(`Frontend server running on port ${FRONTEND_PORT}`);
            resolve();
        });
        expressServer.on('error', (error) => {
            log(`Failed to start frontend server: ${error.message}`, 'ERROR');
            reject(error);
        });
    });
}
// 메인 윈도우 생성
function createWindow() {
    log('Creating main window...');
    mainWindow = new electron_1.BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
        },
        title: APP_NAME,
    });
    const url = electron_is_dev_1.default
        ? `http://localhost:3000` // 개발 모드: React dev server
        : `http://localhost:${FRONTEND_PORT}`; // 프로덕션: Express server
    log(`Loading URL: ${url}`);
    mainWindow.loadURL(url);
    if (electron_is_dev_1.default) {
        mainWindow.webContents.openDevTools();
    }
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
    mainWindow.on('ready-to-show', () => {
        log('Window ready to show');
        mainWindow?.show();
    });
}
// 앱 초기화
async function initialize() {
    try {
        log(`${APP_NAME} starting...`);
        log(`Mode: ${electron_is_dev_1.default ? 'Development' : 'Production'}`);
        log(`User data path: ${userDataPath}`);
        // DB 초기화
        initializeDatabase();
        // 백엔드 시작
        await startBackend();
        // 프론트엔드 시작 (프로덕션만)
        if (!electron_is_dev_1.default) {
            await startFrontend();
        }
        // 윈도우 생성
        createWindow();
        log('Application initialized successfully');
    }
    catch (error) {
        log(`Initialization failed: ${error}`, 'ERROR');
        electron_1.app.quit();
    }
}
// 앱 이벤트 처리
electron_1.app.on('ready', initialize);
electron_1.app.on('window-all-closed', () => {
    log('All windows closed');
    // 백엔드 프로세스 종료
    if (backendProcess) {
        log('Killing backend process...');
        backendProcess.kill();
    }
    // Express 서버 종료
    if (expressServer) {
        log('Closing frontend server...');
        expressServer.close();
    }
    electron_1.app.quit();
});
electron_1.app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});
// IPC 핸들러
electron_1.ipcMain.on('get-log-path', (event) => {
    event.reply('log-path', logFile);
});
// 종료 전 정리
electron_1.app.on('before-quit', () => {
    log('Application quitting...');
});
