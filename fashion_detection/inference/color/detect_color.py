from color.kmeans import k_means
import cv2
def get_color(img,cord):
    img = img[cord[1]:cord[3],cord[0]:cord[2]]
    img = cv2.resize(img,(300,300))
    k_mean_image = k_means(img).process(1)
    return k_mean_image