import unittest
from PIL import Image
import os
import shutil
from main import app, process_image, segment_image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class TestImageProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a test uploads directory
        cls.test_uploads_dir = 'test_uploads'
        if not os.path.exists(cls.test_uploads_dir):
            os.makedirs(cls.test_uploads_dir)

        # Create a sample image for testing
        cls.sample_image_path = os.path.join(cls.test_uploads_dir, 'sample_image.jpg')
        image = Image.new('RGB', (100, 100), color = 'red')
        image.save(cls.sample_image_path)

    @classmethod
    def tearDownClass(cls):
        # Remove the test uploads directory after tests
        if os.path.exists(cls.test_uploads_dir):
            shutil.rmtree(cls.test_uploads_dir)

    def test_process_image(self):
        # Test the process_image function
        processed_image_filename = process_image(self.sample_image_path, 'humans')
        processed_image_path = os.path.join(self.test_uploads_dir, processed_image_filename)
        print(f"Processed image path: {processed_image_path}")
        self.assertTrue(os.path.exists(processed_image_path))

    def test_segment_image(self):
        # Test the segment_image function
        segmented_image_filename = segment_image(self.sample_image_path, 'humans')
        segmented_image_path = os.path.join(self.test_uploads_dir, segmented_image_filename)
        print(f"Segmented image path: {segmented_image_path}")
        self.assertTrue(os.path.exists(segmented_image_path))

    def test_upload_route(self):
        # Test the upload route
        with app.test_client() as client:
            data = {
                'image': (open(self.sample_image_path, 'rb'), 'sample_image.jpg'),
                'action': 'humans'
            }
            response = client.post('/upload', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 302)  # Check for redirect

    def test_results_route(self):
        # Test the results route
        with app.test_client() as client:
            response = client.get('/results?filename=sample_image.jpg&action=humans')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Image Processing Results', response.data)

if __name__ == '__main__':
    unittest.main()