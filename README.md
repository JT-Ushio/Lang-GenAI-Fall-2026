# 以语言为中心的生成式AI

2026 年秋季课程仓库。`lecture_00.py` 用于框架测试；`lecture_01.py` 为第一讲导论初稿（45 分钟 × 2 课时），待教师继续完善。

第一讲：`cd lectures && python3 execute.py -m lecture_01`，浏览 `?trace=var/traces/lecture_01.json`。时间分配、讨论参考与待确认事项见 [第一讲备课说明](lectures/notes/lecture_01.md)。

课件框架复用 [2026 春季《自然语言处理与语言习得》](https://github.com/JT-Ushio/nlp-and-la-spring-2026) 的 Python 执行轨迹生成器和 React Trace Viewer，保留 Markdown、公式、图片、链接、代码与变量展示、逐步浏览等功能。

## 安装与运行

需要 Python 3.10+、Node.js 22.12+（本地验证使用 Python 3.13、Node.js 24）。在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd lectures/trace-viewer
npm ci
cd ../..
```

编译模拟课件：

```bash
source .venv/bin/activate
cd lectures
python execute.py -m lecture_00
cd trace-viewer
npm run dev -- --host 127.0.0.1
```

打开终端提示的地址，默认加载 `lecture_00`；也可访问
[模拟课件](http://localhost:5173/?trace=var/traces/lecture_00.json)。

## 编写课件

在 `lectures/` 下新建 `lecture_01.py` 等文件，定义 `main()`，参考 `lecture_00.py` 使用 `text()`、`image()`、`link()`。本地图片放入 `lectures/images/`；变量所在行添加 `# @inspect variable_name` 可显示运行值。

在 `lectures/` 目录执行 `python execute.py -m lecture_01`，再通过 `?trace=var/traces/lecture_01.json` 浏览。每次修改 Python 课件后需要重新编译；静态网站不会执行 Python。

左右方向键逐步浏览，Shift + 左右方向键跳过函数，大写 R（Shift + R）切换源码，大写 A（Shift + A）切换逐步展示。

## 视频

支持本地视频与 HTTP(S) 视频直链（浏览器原生播放器），以及 B站 BV 链接（官方内嵌播放器）：

```python
from execute_util import video

video("videos/demo.mp4", width=800)
video("https://example.com/demo.mp4", width="100%")
video("https://www.bilibili.com/video/BV1CJbc6JExi/", width=800)
```

本地文件放在 `lectures/videos/`，路径相对于 `lectures/`。框架通过 `public/videos` 符号链接在开发和静态构建时提供这些文件。`lecture_00.py` 包含一个本地生成的 6 秒测试视频。

默认不自动播放，支持进度条、音量及全屏；切换步骤或逐步展示模式时暂停并保留播放位置；进入源码模式时停止播放，切回课件模式后重新加载视频。焦点在播放器上时，方向键交给播放器处理。

普通在线直链必须直接返回浏览器可播放的视频内容，不能使用 YouTube 等网页地址。B站视频页链接单独识别。在线视频不会在编译时下载或打包，播放时需要联网，且受源站访问权限、链接有效期及防盗链限制。建议使用 MP4（H.264 视频、AAC 音频）；加载失败时会显示提示和原始视频链接。

### B站视频

直接传入完整的 `https://www.bilibili.com/video/BV.../` 链接即可。自动移除追踪参数，支持 `?p=2` 选择分集。暂不解析 `b23.tv` 短链接。

点击“加载 B站播放器”后，再点击播放器中的播放按钮。支持 B站提供的全屏、音量等控件；始终提供“在 B站打开”链接作为备用。播放是否可用及清晰度受 B站网络、登录和视频权限限制。

切换步骤、展示模式或进入源码模式会卸载 B站播放器，停止后台声音；返回后需重新加载，不保留播放进度。`lecture_00.py` 已加入所提供的三个视频。

## 构建静态版本

先编译需要展示的课件，再运行：

```bash
cd lectures/trace-viewer
npm run build
npm run preview -- --host 127.0.0.1
```

产物位于 `lectures/trace-viewer/dist/`，包含编译后的 JSON 与图片，可放到静态服务器的根目录或子目录。`public/images` 和 `public/var` 是指向课件资源的相对符号链接。数学公式沿用春季框架的 MathJax CDN，需要联网加载；已修复 CDN 尚未就绪时导致页面白屏的时序问题。

## 来源

迁移源版本：`nlp-and-la-spring-2026` 的 `e6dcd5a7a4b7ae81425db2912b9db064e575f568`。仅迁移通用框架，不包含春季讲义、试卷、课程图片或已安装的 `node_modules`。

原框架受 [Stanford CS336](https://github.com/stanford-cs336/spring2025-lectures) 启发。许可证见 [LICENSE](LICENSE)。
