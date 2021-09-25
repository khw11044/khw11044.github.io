---
layout: post
bigtitle:  "Visual Studio Code에 Anaconda 연동"
subtitle:   "VS Code Terminal을 conda 로 시작"
categories:
    - blog
    - blog-etc
tags:
    - pose
comments: true
published: true
---

## visual studio code 기본 터미널 Anaconda로 하기

이번에 VS code를 새로 설치하면 이전과 다른 세팅으로 해야한다.

<https://khw11044.github.io/blog/blog-etc/2020-12-21-setting-start/>

이전에 포스팅한 VS code 기본 터미널을 anaconda (base)로 시작하는 세팅인데

File > Preference > Settings에  terminal.integrated.shellArgs.windows 가 없다.

열심히 찾아 봤는데 결론만 말하자면 Terminal › Integrated › Automation Shell: Windows 로 바뀌었고

여기서 Edit in settings.json을 보면 있는 코드는 다 지워주고 아래 코드를 넣어주면 된다.

~~~pytho
{
    "python.pythonPath": "C:\\Anaconda3\\python.exe",
    "terminal.integrated.shell.windows": "C:\\Windows\\System32\\cmd.exe",
    "terminal.integrated.shellArgs.windows": ["/K", "C:\\Anaconda3\\Scripts\\activate.bat C:\\Anaconda3"
    ],
    "git.autofetch": true,
    "git.enableSmartCommit": true,
    "workbench.editorAssociations": {
        "*.ipynb": "jupyter-notebook"
    },
    "kite.showWelcomeNotificationOnStartup": false,
    "workbench.startupEditor": "none",
    "notebook.cellToolbarLocation": {
        "default": "right",
        "jupyter-notebook": "left"
    },
    "python.defaultInterpreterPath": "C:\\Anaconda3\\python.exe",
    "terminal.integrated.automationShell.windows": ""
}
~~~

물론 나는 Anaconda install시 C:\\\Anaconda3 에 위치시켰다