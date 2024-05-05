import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from glob import glob


def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result, image_framed


def show_result(result):
    root = tk.Tk()
    root.title("OCR文字识别结果")
    root.geometry("400x300")

    scrollbar = tk.Scrollbar(root)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_area = tk.Text(root, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text_area.pack(expand=True, fill=tk.BOTH)

    scrollbar.config(command=text_area.yview)

    for key in result:
        text_area.insert(tk.END, result[key][1] + '\n')

    root.mainloop()


if __name__ == '__main__':
    image_files = glob('./test_images/1.png')
    result_dir = './test_result'
    for image_file in sorted(image_files):
        t = time.time()
        result, image_framed = single_pic_proc(image_file)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        txt_file = os.path.join(result_dir, image_file.split('/')[-1].split('.')[0] + '.txt')
        #
        print(txt_file)
        txt_f = open(txt_file, 'w')
        Image.fromarray(image_framed).save(output_file)
        print("任务完成，耗时 {:.3f} 秒".format(time.time() - t))
        print("\n识别结果:\n")
        for key in result:
            print(result[key][1])
            txt_f.write(result[key][1] + '\n')
        txt_f.close()

        show_result(result)

