import cv2
import pytesseract

# # 设置Tesseract OCR路径（根据你的安装路径进行调整）
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
# 加载图片
image = cv2.imread('ticket/20240105213742149_1_0.jpg')

# 定义滑动窗口大小和步长
window_size = (100, 100)  # 窗口大小
step_size = 50  # 步长

# 在图像上滑动窗口
for y in range(0, image.shape[0], step_size):
    for x in range(0, image.shape[1], step_size):
        # 截取当前窗口
        window = image[y:y + window_size[1], x:x + window_size[0]]

        # 使用Tesseract进行文本识别
        text = pytesseract.image_to_string(window, lang='eng')

        # 打印识别的文本
        print(f"Text at position ({x}, {y}):\n{text}")

        # 可以根据需要将识别的文本保存到文件中或进行其他处理


