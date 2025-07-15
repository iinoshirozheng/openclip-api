# 步驟 1: 指定正確的基礎映像檔來源
# 我們從 GitHub Container Registry (ghcr.io) 拉取官方的 pixi 映像檔
FROM ghcr.io/prefix-dev/pixi:latest

# 步驟 2: 設定工作目錄
# 在容器內建立一個名為 /app 的資料夾，並將其設定為後續指令的執行目錄
WORKDIR /app

# 步驟 3: 複製專案檔案
# 將你本機的專案檔案（Dockerfile 所在的目錄下所有檔案）複製到容器的 /app 目錄中
COPY . .

# 說明：通常下一步是安裝專案依賴
# 如果你的專案有 pixi.toml 檔案，你可以在這裡加上安裝指令
RUN pixi install

# 說明：最後，定義容器啟動時要執行的預設命令
CMD ["pixi", "run", "serve"]
