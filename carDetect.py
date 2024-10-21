import cv2
import numpy as np
from skimage import measure
import imutils
from collections import deque
import tensorflow as tf

def sort_cont(character_contours):
    
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes), key=lambda b: b[1][i], reverse=False))
    return character_contours

def segment_chars(plate_img, fixed_width):
   
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype='uint8')
    characters = []

    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = bgr_thresh[y:y + h + (addPixel * 2), x:x + w + (addPixel * 2)]
            characters.append(temp)

    return characters if characters else None

class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (5, 5), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morph_img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, element)
        return morph_img

    def extract_contours(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)

            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None

            return plate, True, [x, y, w, h]

        return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)

            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if characters_on_plate is not None and len(characters_on_plate) >= 6:  # Allowing for more flexibility
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    after_check_plate_img = after_clean_plate_img
                    return after_check_plate_img, characters_on_plate, coordinates

        return None, None, None

    def find_possible_plates(self, input_img):
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []
        scales = [1.0, 1.5, 0.75]

        for scale in scales:
            scaled_img = cv2.resize(input_img, None, fx=scale, fy=scale)
            self.after_preprocess = self.preprocess(scaled_img)
            possible_plate_contours = self.extract_contours(self.after_preprocess)

            for cnts in possible_plate_contours:
                plate, characters_on_plate, coordinates = self.check_plate(scaled_img, cnts)
                if plate is not None:
                    plates.append(plate)
                    self.char_on_plate.append(characters_on_plate)
                    self.corresponding_area.append((coordinates[0] / scale, coordinates[1] / scale))

        return plates if plates else None

    def find_characters_on_plate(self, plate):
        charactersFound = segment_chars(plate, 400)
        return charactersFound if charactersFound else None

    def ratioCheck(self, area, width, height):
        min_area = self.min_area
        max_area = self.max_area
        ratioMin = 2
        ratioMax = 7
        ratio = float(width) / float(height)

        if ratio < 1:
            ratio = 1 / ratio

        if (area < min_area or area > max_area) or (ratio < ratioMin or ratio > ratioMax):
            return False

        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        if (width > height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle

        if angle > 25:
            return False

        if (height == 0 or width == 0):
            return False

        area = width * height
        return self.ratioCheck(area, width, height)

class OCR:
    def __init__(self, modelFile, labelFile):
        self.model_file = modelFile
        self.label_file = labelFile
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto())

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
            with graph.as_default():
                tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        """Transform an image into tensor."""
        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):
        input_name = "import/input"
        output_name = "import/final_result"
        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)
        results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            plate += self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)

if __name__ == "__main__":
    findPlate = PlateFinder(minPlateArea=3000, maxPlateArea=20000)
    model = OCR(modelFile="./binary_128_0.50_ver3.pb", labelFile="./binary_128_0.50_labels_ver2.txt")
    cap = cv2.VideoCapture('./test1.MOV')
    recent_detections = deque(maxlen=10)

    while (cap.isOpened()):
        ret, img = cap.read()
        if ret:
            cv2.imshow('original video', img)
            possible_plates = findPlate.find_possible_plates(img)
            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                    recent_detections.append(recognized_plate)
                    unique_detections = set(recent_detections)
                    print("Recent unique detections:", unique_detections)
                    cv2.imshow('plate', p)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
