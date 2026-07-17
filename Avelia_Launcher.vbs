' ==== Avelia Silent Launcher ====

Set WshShell = CreateObject("WScript.Shell")

projectPath = "C:\Mia_project\mia_2"
pythonwPath = "C:\Program Files\Python\pythonw.exe"

WshShell.CurrentDirectory = projectPath
WshShell.Run """" & pythonwPath & """ main.py", 0, False