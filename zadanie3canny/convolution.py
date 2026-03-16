def convolve(img, kernel):
    res = img.copy()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            value = 0
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    try:
                        value += img[x - 2 + i][y - 2 + j] * kernel[i][j]
                    except:
                        pass
            res[x, y] = int(value)
    return res