import { contextBridge, ipcRenderer } from 'electron';

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electron', {
  getLogPath: () => {
    ipcRenderer.send('get-log-path');
    return new Promise((resolve) => {
      ipcRenderer.once('log-path', (event, logPath) => {
        resolve(logPath);
      });
    });
  },
});

// Type declaration for TypeScript
declare global {
  interface Window {
    electron: {
      getLogPath: () => Promise<string>;
    };
  }
}
