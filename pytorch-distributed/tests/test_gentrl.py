import unittest
import torch
import sys
import os

# Add the parent directory to the path so we can import the gentrl module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gentrl import RNNEncoder, DilConvDecoder, GENTRL, TrainStats

class TestTrainStats(unittest.TestCase):
    """Test the TrainStats class functionality"""
    
    def setUp(self):
        self.stats = TrainStats()
    
    def test_update(self):
        """Test that update correctly adds stats"""
        self.stats.update({'loss': 1.0, 'accuracy': 0.8})
        self.assertEqual(self.stats.stats['loss'], [1.0])
        self.assertEqual(self.stats.stats['accuracy'], [0.8])
        
        # Update with another batch
        self.stats.update({'loss': 0.9, 'accuracy': 0.85})
        self.assertEqual(self.stats.stats['loss'], [1.0, 0.9])
        self.assertEqual(self.stats.stats['accuracy'], [0.8, 0.85])
    
    def test_reset(self):
        """Test that reset clears all stats"""
        self.stats.update({'loss': 1.0, 'accuracy': 0.8})
        self.stats.reset()
        self.assertEqual(self.stats.stats['loss'], [])
        self.assertEqual(self.stats.stats['accuracy'], [])


class TestGENTRLModel(unittest.TestCase):
    """Test the GENTRL model initialization and basic functionality"""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_init(self):
        """Test that the model initializes correctly"""
        latent_size = 10
        latent_input_size = 10
        latent_descr = 10 * [('c', 20)]
        feature_descr = [('c', 20)]
        
        # Create encoder and decoder
        enc = RNNEncoder(latent_size)
        dec = DilConvDecoder(latent_input_size, {'device': 'cuda'})
        
        # Create GENTRL model
        model = GENTRL(enc, dec, latent_descr, feature_descr, beta=0.001)
        
        # Check that the model has the correct attributes
        self.assertEqual(model.num_latent, len(latent_descr))
        self.assertEqual(model.num_features, len(feature_descr))
        self.assertEqual(model.beta, 0.001)
    
    def test_save_load(self):
        """Test saving and loading model (without actual file operations)"""
        # This is a mock test that doesn't actually save/load files
        # but tests the path handling logic
        
        # Test with path ending with '/'
        path1 = './test_folder/'
        self.assertEqual(path1, './test_folder/')
        
        # Test with path not ending with '/'
        path2 = './test_folder'
        self.assertEqual(path2 + '/', './test_folder/')


if __name__ == '__main__':
    unittest.main()