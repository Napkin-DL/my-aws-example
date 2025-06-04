import unittest
import sys
import os

# Add the parent directory to the path so we can import the gentrl module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestQuestion(unittest.TestCase):
    """
    Test class specifically created for the 'test-question test' request.
    This demonstrates a simple test case that always passes.
    """
    
    def test_question(self):
        """Test that the test question passes"""
        self.assertTrue(True, "This test should always pass")
    
    def test_addition(self):
        """Test basic addition"""
        self.assertEqual(1 + 1, 2, "1 + 1 should equal 2")
    
    def test_string_concatenation(self):
        """Test string concatenation"""
        self.assertEqual("test" + "-" + "question", "test-question", 
                         "String concatenation should work correctly")


if __name__ == '__main__':
    unittest.main()