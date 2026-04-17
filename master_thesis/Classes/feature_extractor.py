#DEFAULT
import os
import gzip
import os
import json
#3RD PARTY
import cv2
import pytesseract
import pandas as pd
import numpy as np
from skimage.restoration import estimate_sigma
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from ultralytics import YOLO
import langdetect
import urllib
#CUSTOM


class FeatureExtractor:
    def __init__(self):
        self.features = dict()
        self.normalized_features = dict()
        # Min max values to normalize features based on the dataset already learned by the tabular model
        self.min_max_values = json.load(open("../../data/training/normalization_min_max_values.json", "r"))

    def extract_ocr_based_features(self, image_path):
        img = cv2.imread(image_path)
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
        ocr_data = ocr_data[ocr_data.conf != -1] if not ocr_data.empty else ocr_data # remove empty rows
        ocr_data = ocr_data[ocr_data.text.str.strip() != ''] if not ocr_data.empty else ocr_data 
        if ocr_data.empty:
            return dict(
                mean_conf=0.0,
                min_conf=0.0,
                std_conf=0.0,
                words_count=0,
                char_count=0,
                num_text_lines=0,
                num_text_blocks=0,
                low_conf_ratio=0.0,
                text_density=0.0,
                handwriting_ocr=0.0,
                detected_lang='zz',
                detected_lang_conf=0.0
            )
        mean_conf = ocr_data.conf.mean()
        min_conf = ocr_data.conf.min()
        std_conf = ocr_data.conf.std()
        words_count = len(ocr_data)
        char_count = ocr_data['text'].str.len().sum()
        num_text_lines = sum([max(line_numbers) for _, line_numbers in ocr_data.groupby('block_num')['line_num']])
        num_text_blocks = ocr_data['block_num'].nunique()
        low_conf_ratio = (ocr_data.conf < 50).sum() / words_count # ratio of words with very bad confidence below 50
        text_density = (ocr_data.width * ocr_data.height).sum() / img.shape[0] * img.shape[1] # ratio of text area to image area
        handwriting_ocr = 1.0 if mean_conf < 50 and text_density > 0.05 else 0.0
        detected_lang, detected_lang_conf = self._get_languages_from_ocr(ocr_data)
        return dict(
            mean_conf=mean_conf,
            min_conf=min_conf,
            std_conf=std_conf,
            words_count=words_count,
            char_count=char_count,
            num_text_lines=num_text_lines,
            num_text_blocks=num_text_blocks,
            low_conf_ratio=low_conf_ratio,
            text_density=text_density,
            handwriting_ocr=handwriting_ocr,
            detected_lang=detected_lang,
            detected_lang_conf=detected_lang_conf
        )
        
    def extract_visual_features(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_image_area = img.shape[0] * img.shape[1]
        relevant_countours = 0
        text_area = 0
        label_coordinates = []
        mean_orientation = 0
        x_min, y_min = img.shape[1], img.shape[0]
        x_max, y_max = 0, 0

        # heuristic size measurements
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 400:
                continue
            if area > 0.075 * total_image_area:
                continue
            if w / float(h) < 1.2:
                continue
            cy = y + h / 2.0
            if cy < img.shape[0] * 2 / 3:
                continue
            if w > 20 and h > 10: 
                relevant_countours += 1
                text_area += w * h
                label_coordinates.append((x, y, w, h))
                mean_orientation += np.arctan2(h, w)
                x_min = x if x < x_min else x_min
                x_max = x + w if x + w > x_max else x_max
                y_min = y if y < y_min else y_min
                y_max = y + h if y + h > y_max else y_max

        text_density = text_area / total_image_area
        mean_orientation = mean_orientation / max(relevant_countours, 1)
        relative_label_centroid_x = (x_min + x_max) / 2.0 / img.shape[1]
        relative_label_centroid_y = (y_min + y_max) / 2.0 / img.shape[0]

        return dict(
            text_density=text_density,
            text_area=text_area,
            mean_orientation=mean_orientation,
            relative_label_centroid_x=relative_label_centroid_x,
            relative_label_centroid_y=relative_label_centroid_y
        )
    
    def extract_label_features(self, images_folder_path):
        label_features_dict = dict()
        labels_dict = self._identify_labels_using_yolo(images_folder_path)
        labels_dict = dict(list(labels_dict.items()))
        for image, boxes in labels_dict.items():
            number_of_labels = len(boxes)
            label_coordinates = boxes
            mean_label_width = 0.0
            mean_label_height = 0.0
            areas = 0.0
            centroids_x = []
            centroids_y = []
            for box in boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                mean_label_width += width
                mean_label_height += height
                areas += width * height
                centroids_x.append(x1 + x2 / 2.0)
                centroids_y.append(y1 + y2 / 2.0)
            if number_of_labels > 0:
                mean_label_width = mean_label_width / number_of_labels
                mean_label_height = mean_label_height / number_of_labels
            else:
                mean_label_width = 0.0
                mean_label_height = 0.0
            spread_centroids_x = np.std(centroids_x) if len(centroids_x) > 1 else 0.0
            spread_centroids_y = np.std(centroids_y) if len(centroids_y) > 1 else 0.0
            label_spread = np.sqrt(spread_centroids_x ** 2 + spread_centroids_y ** 2)
            handwritten_flag = 0.0 # TODO: Use a classifier to determine if handwritten or not
            label_features_dict[image.split("\\")[-1]] = {
                'number_of_labels': number_of_labels,
                'label_coordinates': label_coordinates,
                'mean_label_width': mean_label_width,
                'mean_label_height': mean_label_height,
                'label_spread': label_spread,
                'areas': areas,
                'handwritten_flag': handwritten_flag
            }
            
        return label_features_dict
    
    def extract_image_quality_features(self, input_image_path, label_features=None):
        input_image = cv2.imread(input_image_path)
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        resolution = input_image.shape[1] * input_image.shape[0] / 1000000  # in megapixels
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        brightness = gray.mean()
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21) # quite slow
        noise = np.mean((gray - noise) ** 2)
        noise_sigma = np.mean(estimate_sigma(gray, average_sigmas=True))
        background_uniformity = self._calculate_background_heterogeneity(gray)
        overlapped_label_area = self._check_for_overlaps(input_image_path, label_features)
        image_quality_features = dict(
            resolution=resolution,
            laplacian=laplacian,
            contrast=contrast,
            brightness=brightness,
            noise=noise,
            noise_sigma=noise_sigma,
            background_uniformity=background_uniformity,
            overlapped_label_area=overlapped_label_area # TODO: Should be in label feature extraction
        )
        return image_quality_features
    
    def extract_features(self, images_folder_path):
        allowed_extensions = ['jpg', 'jpeg', 'png']
        image_files = []
        for image_file in os.listdir(images_folder_path):
            if image_file.rsplit('.', 1)[-1].lower() in allowed_extensions:
                image_files.append(os.path.join(images_folder_path, image_file))
        label_features = self.extract_label_features(images_folder_path)
        for i in range(len(image_files)):
            image_file = image_files[i]
            ocr_features = self.extract_ocr_based_features(image_file)
            visual_features = self.extract_visual_features(image_file)
            image_quality_features = self.extract_image_quality_features(image_file, label_features=label_features)
            self.features[image_file.split("/")[-1]] = {**ocr_features, **visual_features, **image_quality_features, **label_features.get(image_file.split("/")[-1], {})}

        features_df = pd.DataFrame.from_dict(self.features, orient='index')
        self.normalized_features = self._fit(features_df)
        self.normalized_features.to_csv("../../data/sample_images_ood/extracted_features_ood.csv", index=True)

        return self.normalized_features
            
    def _fit(self, features):
        # method for normalizing the feature value ranges to 0-1 
        imputer = SimpleImputer(strategy='median')

        numeric_features = features.columns[features.dtypes != 'object']
        detected_languages = self.min_max_values['detected_lang'][0]
        if detected_languages:
            detected_languages_dict = {}
            i = 0
            for lang in detected_languages:
                detected_languages_dict[lang] = i
                i += 1
            features['detected_lang_encoded'] = 0
            for row in features.itertuples():
                lang = row.detected_lang
                if lang in detected_languages_dict:
                    features.at[row.Index, 'detected_lang_encoded'] = detected_languages_dict[lang]
        self.min_max_values['detected_lang_encoded'] = [0, len(detected_languages) - 1]
                
        numeric_features = features.columns[features.dtypes != 'object']

        features[numeric_features] = imputer.fit_transform(features[numeric_features])
        
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        features_for_scaling = features[numeric_features].copy()
        if 'detected_lang_encoded' not in self.min_max_values:
            self.min_max_values['detected_lang_encoded'] = features['detected_lang_encoded']
        for col in numeric_features:
            min_max_scaler.fit(pd.DataFrame(self.min_max_values[col], columns=[col]))
            features[col] = features[col].clip(self.min_max_values[col][0], self.min_max_values[col][1])
            features_for_scaling[col] = min_max_scaler.transform(features[[col]])

        return features_for_scaling

    
    def _get_languages_from_ocr(self, ocr_data):
        langdetect.DetectorFactory.seed = 0
        languages = dict()
        for block in ocr_data['block_num'].unique():
            ocr_data_block = ocr_data[ocr_data['block_num'] == block].dropna(subset=['text', 'conf'])
            block_text = ' '.join(ocr_data_block['text'].tolist())
            block_confidence = ocr_data_block['conf'].mean()
            if len(block_text.strip()) < 25 or block_confidence < 50 or 'museum' in block_text.lower() or 'herbarium' in block_text.lower() or \
            'herb.' in block_text.lower() or 'collection' in block_text.lower():
                continue
            try:
                lang = langdetect.detect_langs(block_text)
                language = str(lang[0]).split(':')[0]
                confidence = float(str(lang[0]).split(':')[1])
                if language not in languages:
                    languages[language] = {'conf': confidence, 'char_count': len(block_text), 'count': 1}
                else:
                    languages[language]['conf'] += confidence
                    languages[language]['char_count'] += len(block_text)
                    languages[language]['count'] += 1
            except:
                continue
        if len(languages) == 0:
            return 'zz', 0.0
        else:
            lang_values = dict()
            for language in languages:
                lang_values[language] = (languages[language]['conf'] / languages[language]['count'] * languages[language]['char_count'])
            detected_lang = max(lang_values, key=lang_values.get)
            return detected_lang, lang_values[detected_lang]
        
    def _calculate_background_heterogeneity(self, gray_image):
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        background = gray_image[thresh == 255]
        if len(background) == 0:
            return 0.0
        return np.std(background)

    def _check_for_overlaps(self, input_image_path, label_features=None):
        # Match YOLO-detected labels with visual detection to find overlaps
        # extract institutional label paths from hespi results for the given image
        # TODO: Check also for B101126624.handwritten_data.jpg
        input_image = cv2.imread(input_image_path)
        gray = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_coords = []

        for cnt in contours:
            contour_shape = cv2.approxPolyDP(cnt, 3, True)
            cv2.drawContours(input_image, [contour_shape], -1, (0, 255, 0), 2)
            # From numpy.org/doc/2.2/reference/generated/numpy.ravel.html
            contours_coords.append(contour_shape.ravel().tolist()) # flatten to list of x1,y1,x2,y2,...
        id = input_image_path.rsplit("/", 1)[-1]

        label_images = []

        label_images = self.features[id]['label_coordinates'] if id in self.features and 'label_coordinates' in self.features[id] else []
        overlap_ratio = np.float64(0.0)
        if type(label_images) == str:
            label_images = eval(label_images)
        if len(label_images) == 0:
            return overlap_ratio
        for label_image_coords in label_images:
            x1, y1, x2, y2 = label_image_coords  # unpack coordinates
            max_loc = (x1, y1)
            top_left = max_loc
            h = y2 - y1
            w = x2 - x1
            bottom_right = (top_left[0] + w, top_left[1] + h)

            for cnt_coords in contours_coords:
                if top_left[0] <= min(cnt_coords[0::2]) and bottom_right[0] >= max(cnt_coords[0::2]) and \
                    top_left[1] <= min(cnt_coords[1::2]) and bottom_right[1] >= max(cnt_coords[1::2]):
                    continue # matched region completely contains contour
                if min(cnt_coords[0::2]) <= top_left[0] and max(cnt_coords[0::2]) >= bottom_right[0] and \
                    min(cnt_coords[1::2]) <= top_left[1] and max(cnt_coords[1::2]) >= bottom_right[1]:
                    continue # contour completely contains matched region
                for i in range(0, len(cnt_coords), 2):
                    x, y = cnt_coords[i], cnt_coords[i+1]
                    if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                        # calculate overlap area
                        area = (min(bottom_right[0], max(cnt_coords[0::2])) - max(top_left[0], min(cnt_coords[0::2]))) * \
                                (min(bottom_right[1], max(cnt_coords[1::2])) - max(top_left[1], min(cnt_coords[1::2])))
                        if area / (w * h) < 0.5:
                            overlap_ratio += area / (w * h)
                            break
        return overlap_ratio
    
    def _identify_labels_using_yolo(self, images_folder_path):
        # Weights from HESPI's institutional label detector
        from ultralytics.models.yolo.detect.predict import DetectionPredictor

        weight_url = 'http://github.com/rbturnbull/hespi/releases/download/v0.4.0/sheet-component.pt.gz'
        weights_path_compressed = 'sheet-component.pt.gz'
        weights_path = 'sheet-component.pt'
        if not os.path.exists(weights_path):
            
            urllib.request.urlretrieve(weight_url, weights_path_compressed)
            with gzip.open(weights_path_compressed, 'rb') as file_in:
                with open(weights_path, 'wb') as file_out:
                    file_out.write(file_in.read())
            os.remove(weights_path_compressed)
        model = YOLO(weights_path, verbose=False)
        model.to('cpu')

        os.makedirs('temp_labels', exist_ok=True) # should be created to provide bboxes to classifier

        if not model.predictor:
            model.predictor = DetectionPredictor()
            
        model.predictor.setup_model(model=model.model)
        
        results = model.predict(source=images_folder_path, show=False, save=False, imgsz=1280, stream=True, verbose=False)

        label_dict = dict()

        for res in results:
            image = res.path
            if image not in label_dict:
                label_dict[image] = []
            sheet_boxes = res.boxes if res and len(res) > 0 else []
            for box in sheet_boxes:
                if box.cls == 9: # institutional label class
                    x1, y1, x2, y2 = box.xyxy[0]
                    label_dict[image].append((int(x1), int(y1), int(x2), int(y2)))
            
        return label_dict