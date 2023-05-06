from PIL import Image

if __name__ == '__main__':
    img = Image.open('./handwritten/0/1.jpg')
    width, height = img.size

    output_img = Image.new('RGB', (width * 5, height * 10))

    for i in range(10):
        for j in range(5):
            img = Image.open(f'./handwritten/{i}/{j + 1}.jpg')
            output_img.paste(img, (j * width, i * height))

    output_img.save('./handwritten/handwrittenlist.jpg')
