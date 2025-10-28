"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
electron_1.contextBridge.exposeInMainWorld('electron', {
    getLogPath: () => {
        electron_1.ipcRenderer.send('get-log-path');
        return new Promise((resolve) => {
            electron_1.ipcRenderer.once('log-path', (event, logPath) => {
                resolve(logPath);
            });
        });
    },
});
