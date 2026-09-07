"""框架测试课件；正式授课内容由教师后续编写。"""

from execute_util import image, link, text, video


def main():
    text("# 以语言为中心的生成式AI")
    text("## 2026 秋季 · Lecture 00：框架测试")
    text("这是一份模拟课件，用于检查编译与展示，不代表正式课程安排。")
    text("### 文字、公式与图片")
    text("支持 **Markdown**、中文、English，以及公式：$p(x)=\\prod_{t=1}^{T}p(x_t\\mid x_{<t})$。")
    image("images/lecture_00.svg", width=640)
    link(title="课程仓库", url="https://github.com/JT-Ushio/Lang-GenAI-Fall-2026")
    token_demo()
    text("### 本地视频测试")
    video("videos/lecture_00.mp4", width=640)
    text("点击播放器播放；切换课件步骤时自动暂停。")
    text("### B站视频 1")
    video("https://www.bilibili.com/video/BV1CJbc6JExi/", width=800)
    text("### B站视频 2")
    video("https://www.bilibili.com/video/BV156KL6XESt/", width=800)
    text("### B站视频 3")
    video("https://www.bilibili.com/video/BV1xF3f6qEJ9/", width=800)
    text("### 测试结束")
    text("使用左右方向键逐步浏览；按 R 切换源码，按 A 切换逐步展示。")


def token_demo():
    text("### 模拟代码：观察变量")
    tokens = ["语言", "与", "生成式AI"]  # @inspect tokens
    token_count = len(tokens)  # @inspect token_count
    assert token_count == 3
    text(f"模拟序列包含 {token_count} 个 token。")


if __name__ == "__main__":
    main()
