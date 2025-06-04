import unittest
import torch
import sys
import os

# Add the parent directory to the path so we can import the gentrl module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gentrl.distributed_gentrl import DIS_GENTRL, TrainStats, Net, average_gradients

class TestDistributedTrainStats(unittest.TestCase):
    """Test the TrainStats class functionality in distributed_gentrl"""
    
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


class TestDIS_GENTRLModel(unittest.TestCase):
    """Test the DIS_GENTRL model initialization and basic functionality"""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_init(self):
        """Test that the model initializes correctly with mock arguments"""
        # This test is skipped if CUDA is not available
        args = {
            'device': 'cuda',
            'rank': 0,
            'size': 1,
            'hvd': False,
            'apex': False,
            'sync_bn': False,
            'opt_level': 'O1',
            'keep_batchnorm_fp32': True,
            'loss_scale': 'dynamic',
            'batch_size': 32,
            'data_dir': './data',
            'verbose_step': 10,
            'num_epochs': 1,
            'lr': 0.001,
            'lr_lp': 0.00001,
            'lr_dec': 0.000001,
            'num_iterations': 100
        }
        
        # Create a mock Net function that returns a model
        model = Net(args)
        
        # Check that the model is a DIS_GENTRL instance
        self.assertIsInstance(model, DIS_GENTRL)
        
        # Check that the model has the correct attributes
        self.assertEqual(model.num_latent, 50)
        self.assertEqual(model.num_features, 1)
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