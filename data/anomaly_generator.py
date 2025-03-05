from PIL import Image
import numpy as np
import cv2

class _Base_Anomaly_Generator():
    def __init__(self):
        pass

    def __call__(self, image:Image):

        ### just for examples -- should return a tuple of three elements
        augmented_image = image
        augmented_mask = np.zeros(image.shape[:2], dtype=np.float32)
        augmented_anomaly = 1

        return augmented_image, augmented_mask, augmented_anomaly

class WhiteNoiseGenerator(_Base_Anomaly_Generator):
    def __init__(self, max_try=200):
        super(WhiteNoiseGenerator, self).__init__()

        self.max_try = max_try

    def __call__(self, image:Image):
        processed_image = image.resize((1024, 1024))
        processed_image = cv2.cvtColor(np.asarray(processed_image), cv2.COLOR_RGB2BGR)
        processed_image = np.array(processed_image).astype(np.float32) / 255.0

        augmented_image, anomaly_mask, has_anomaly = self.augment_image_white_noise(processed_image)

        augmented_image = augmented_image * 255.0
        augmented_image = augmented_image.astype(np.uint8)

        augmented_image = Image.fromarray(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        anomaly_mask = Image.fromarray(anomaly_mask[:, :, 0].astype(np.uint8) * 255, mode='L')

        return augmented_image, anomaly_mask, has_anomaly

    def augment_image_white_noise(self, image):

        # generate noise image
        noise_image = np.random.randint(0, 255, size=image.shape).astype(np.float32) / 255.0
        patch_mask = np.zeros(image.shape[:2], dtype=np.float32)

        # generate random mask
        patch_number = np.random.randint(0, 5)
        augmented_image = image

        for i in range(patch_number):
            try_count = 0
            coor_min_dim1 = 0
            coor_min_dim2 = 0

            coor_max_dim1 = 0
            coor_max_dim2 = 0
            while try_count < self.max_try:
                try_count += 1

                patch_dim1 = np.random.randint(image.shape[0] // 40, image.shape[0] // 10)
                patch_dim2 = np.random.randint(image.shape[1] // 40, image.shape[1] // 10)

                center_dim1 = np.random.randint(patch_dim1, image.shape[0] - patch_dim1)
                center_dim2 = np.random.randint(patch_dim2, image.shape[1] - patch_dim2)

                coor_min_dim1 = np.clip(center_dim1 - patch_dim1, 0, image.shape[0])
                coor_min_dim2 = np.clip(center_dim2 - patch_dim2, 0, image.shape[1])

                coor_max_dim1 = np.clip(center_dim1 + patch_dim1, 0, image.shape[0])
                coor_max_dim2 = np.clip(center_dim2 + patch_dim2, 0, image.shape[1])

                break

            patch_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] = 1.0

        augmented_image[patch_mask > 0] = noise_image[patch_mask > 0]

        patch_mask = patch_mask[:, :, np.newaxis]

        if patch_mask.max() > 0:
            has_anomaly = 1.0
        else:
            has_anomaly = 0.0

        return augmented_image, patch_mask, has_anomaly
        # return augmented_image, patch_mask, np.array([has_anomaly], dtype=np.float32)