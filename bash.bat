@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set STREAMLIT_SERVER_HEADLESS=true

:: Clear Streamlit cache
"D:\visors code\python.exe" -m streamlit cache clear

:: Start the app (替换成你的disease-py.py实际路径！！！)
"D:\visors code\python.exe" -m streamlit run "C:\Users\杜金澎\Desktop\新建文件夹\disease-py.py" --server.address 0.0.0.0 --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false

pause