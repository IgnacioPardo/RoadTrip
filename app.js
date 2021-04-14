const { app, BrowserWindow } = require('electron')

app.commandLine.appendSwitch('ignore-certificate-errors', 'true');

function createWindow () {
  // Create the browser window.
  const win = new BrowserWindow({
    width: 1120,
    height: 630,
    center: true,
    resizable: false,
    frame: false,
    movable: true,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#fff',
    title: 'RoadTrip',
    webPreferences: {
      nodeIntegration: true
    }
  })

  // and load the index.html of the app.
  //win.loadFile('index.html')

  win.setAspectRatio(16/9);
  win.loadURL("http://0.0.0.0:5000");
  // Open the DevTools.
  win.webContents.openDevTools()
}

function server () {
  let {PythonShell} = require('python-shell')
  //change scripts to "electron ." on package.json
  PythonShell.run('main.py', null, function (err) {
    if (err) throw err;;
    createWindow();
  });
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(createWindow)
//app.whenReady().then(server)

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
