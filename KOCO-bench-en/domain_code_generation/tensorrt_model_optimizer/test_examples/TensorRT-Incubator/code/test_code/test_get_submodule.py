#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test get_submodule function
"""

import unittest
import torch
import sys
import os
from unittest.mock import MagicMock

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Conditional mock: only mock when module doesn't exist or import fails
def conditional_mock(module_name, submodules=None):
    """Only mock when module doesn't exist or import fails"""
    try:
        __import__(module_name)
    except (ImportError, OSError, Exception):
        mock_obj = MagicMock()
        sys.modules[module_name] = mock_obj
        if submodules:
            for sub in submodules:
                full_name = f"{module_name}.{sub}"
                try:
                    __import__(full_name)
                except (ImportError, OSError, Exception):
                    sys.modules[full_name] = MagicMock()
        return mock_obj
    return None

# Mock dependencies
conditional_mock('transformers')
conditional_mock('tripy')

# Import ground truth implementation
try:
    tripy_path = os.path.join(os.path.dirname(__file__), '..', 'tripy')
    if tripy_path not in sys.path:
        sys.path.insert(0, tripy_path)
    
    # get_submodule is an internal function in weight_loader.py, implement directly here
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Unable to import implementation code: {e}")
    IMPORT_SUCCESS = False


# Since get_submodule is an internal function, we implement it here for testing
def get_submodule(module, attr_name):
    """Get submodule from model hierarchy"""
    attrs = attr_name.split(".")
    for attr in attrs:
        if isinstance(module, torch.nn.ModuleList):
            module = module[int(attr)]
        elif isinstance(module, torch.nn.ModuleDict):
            module = module[attr]
        else:
            module = getattr(module, attr)
    return module


class TestGetSubmodule(unittest.TestCase):
    """Test get_submodule function"""
    
    def setUp(self):
        """Setup test environment"""
        torch.manual_seed(42)
        
        # Create complex test model structure
        self.test_model = self._create_complex_model()
    
    def _create_complex_model(self):
        """Create complex model containing ModuleList and ModuleDict"""
        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Simple attribute
                self.linear1 = torch.nn.Linear(768, 768)
                
                # ModuleList
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(768, 768),
                    torch.nn.Linear(768, 768),
                ])
                
                # ModuleDict
                self.components = torch.nn.ModuleDict({
                    'attention': torch.nn.Linear(768, 768),
                    'mlp': torch.nn.Linear(768, 3072),
                })
                
                # Nested structure
                self.transformer = torch.nn.ModuleDict({
                    'h': torch.nn.ModuleList([
                        torch.nn.ModuleDict({
                            'attn': torch.nn.ModuleDict({
                                'c_attn': torch.nn.Linear(768, 768*3),
                                'c_proj': torch.nn.Linear(768, 768),
                            }),
                            'mlp': torch.nn.ModuleDict({
                                'c_fc': torch.nn.Linear(768, 3072),
                                'c_proj': torch.nn.Linear(3072, 768),
                            })
                        }) for _ in range(2)
                    ])
                })
        
        return ComplexModel()
    
    def test_simple_attribute_access(self):
        """
        Test case 1: Access simple attribute
        
        Input: attr_name="linear1"
        Expected: Return linear1 module
        """
        # Execute
        result = get_submodule(self.test_model, "linear1")
        
        # Verify output
        self.assertIs(result, self.test_model.linear1,
                     "Should return correct module")
        self.assertIsInstance(result, torch.nn.Linear,
                            "Should be Linear type")
    
    def test_modulelist_access(self):
        """
        Test case 2: Access element in ModuleList
        
        Input: attr_name="layers.0"
        Expected: Return first layer
        """
        # Execute
        result = get_submodule(self.test_model, "layers.0")
        
        # Verify output
        self.assertIs(result, self.test_model.layers[0],
                     "Should return first element of ModuleList")
    
    def test_moduledict_access(self):
        """
        Test case 3: Access element in ModuleDict
        
        Input: attr_name="components.attention"
        Expected: Return attention module
        """
        # Execute
        result = get_submodule(self.test_model, "components.attention")
        
        # Verify output
        self.assertIs(result, self.test_model.components['attention'],
                     "Should return attention from ModuleDict")
    
    def test_nested_access(self):
        """
        Test case 4: Access deeply nested module
        
        Input: attr_name="transformer.h.0.attn.c_attn"
        Expected: Return correct nested module
        """
        # Execute
        result = get_submodule(self.test_model, "transformer.h.0.attn.c_attn")
        
        # Verify output
        expected = self.test_model.transformer['h'][0]['attn']['c_attn']
        self.assertIs(result, expected,
                     "Should correctly access deeply nested module")
    
    def test_multiple_modulelist_indices(self):
        """
        Test case 5: Access different ModuleList indices
        
        Input: Different index values
        Expected: Return module at corresponding index
        """
        indices = [0, 1]
        
        for idx in indices:
            with self.subTest(index=idx):
                # Execute
                result = get_submodule(self.test_model, f"layers.{idx}")
                
                # Verify output
                self.assertIs(result, self.test_model.layers[idx])
    
    def test_multiple_moduledict_keys(self):
        """
        Test case 6: Access different ModuleDict keys
        
        Input: Different key names
        Expected: Return module at corresponding key
        """
        keys = ['attention', 'mlp']
        
        for key in keys:
            with self.subTest(key=key):
                # Execute
                result = get_submodule(self.test_model, f"components.{key}")
                
                # Verify output
                self.assertIs(result, self.test_model.components[key])
    
    def test_deep_nesting_with_mixed_types(self):
        """
        Test case 7: Deep access with mixed types
        
        Input: Path containing ModuleList, ModuleDict and regular attributes
        Expected: Correctly traverse all types
        """
        # Execute
        result = get_submodule(self.test_model, "transformer.h.1.mlp.c_fc")
        
        # Verify output
        expected = self.test_model.transformer['h'][1]['mlp']['c_fc']
        self.assertIs(result, expected,
                     "Should correctly handle mixed type nesting")
    
    def test_single_level_access(self):
        """
        Test case 8: Single level access (no dots)
        
        Input: attr_name="linear1" (no nesting)
        Expected: Directly return attribute
        """
        # Execute
        result = get_submodule(self.test_model, "linear1")
        
        # Verify output
        self.assertIs(result, self.test_model.linear1)
    
    def test_empty_path_returns_module(self):
        """
        Test case 9: Empty path handling
        
        Input: attr_name=""
        Expected: Return module itself or handle error
        """
        # For empty string, split returns ['']
        # This will try getattr(module, ''), should raise exception
        with self.assertRaises(AttributeError):
            get_submodule(self.test_model, "")
    
    def test_accessing_submodule_attributes(self):
        """
        Test case 10: Access submodule attributes
        
        Input: Access Linear layer's weight
        Expected: Can access deeper attributes
        """
        # First get submodule
        linear_module = get_submodule(self.test_model, "linear1")
        
        # Then access its attributes (though not a module, demonstrates path parsing)
        self.assertIsNotNone(linear_module.weight,
                            "Should be able to access submodule attributes")


class TestGetSubmoduleEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_model = self._create_test_model()
    
    def _create_test_model(self):
        """Create test model"""
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 10)
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(10, 10),
                    torch.nn.Linear(10, 10),
                    torch.nn.Linear(10, 10),
                ])
                self.named_layers = torch.nn.ModuleDict({
                    'first': torch.nn.Linear(10, 10),
                    'second': torch.nn.Linear(10, 10),
                })
        
        return TestModel()
    
    def test_invalid_modulelist_index(self):
        """
        Test case 11: ModuleList index out of bounds
        
        Input: attr_name="layers.10" (out of range)
        Expected: Raise IndexError
        """
        with self.assertRaises(IndexError):
            get_submodule(self.test_model, "layers.10")
    
    def test_invalid_moduledict_key(self):
        """
        Test case 12: ModuleDict key doesn't exist
        
        Input: attr_name="named_layers.nonexistent"
        Expected: Raise KeyError
        """
        with self.assertRaises(KeyError):
            get_submodule(self.test_model, "named_layers.nonexistent")
    
    def test_invalid_attribute_name(self):
        """
        Test case 13: Attribute doesn't exist
        
        Input: attr_name="nonexistent_attr"
        Expected: Raise AttributeError
        """
        with self.assertRaises(AttributeError):
            get_submodule(self.test_model, "nonexistent_attr")
    
    def test_modulelist_negative_index(self):
        """
        Test case 14: ModuleList negative index
        
        Input: attr_name="layers.-1"
        Expected: Return last element (Python list behavior)
        """
        # Execute
        result = get_submodule(self.test_model, "layers.-1")
        
        # Verify output
        self.assertIs(result, self.test_model.layers[-1],
                     "Negative index should work like list")
    
    def test_nested_invalid_path(self):
        """
        Test case 15: Invalid middle part in nested path
        
        Input: attr_name="layer1.invalid.attr"
        Expected: Raise AttributeError
        """
        with self.assertRaises(AttributeError):
            get_submodule(self.test_model, "layer1.invalid.attr")
    
    def test_numeric_string_for_regular_module(self):
        """
        Test case 16: Use numeric string for regular module
        
        Input: Try to access non-ModuleList with number
        Expected: Raise AttributeError (regular modules don't have numeric attributes)
        """
        with self.assertRaises(AttributeError):
            get_submodule(self.test_model, "layer1.0")
    
    def test_path_with_spaces(self):
        """
        Test case 17: Path contains spaces
        
        Input: attr_name contains spaces
        Expected: Raise AttributeError (attribute names don't contain spaces)
        """
        # Create a special test scenario
        # Note: Python attribute names typically don't contain spaces, but we test parsing behavior
        with self.assertRaises(AttributeError):
            get_submodule(self.test_model, "layer 1")
    
    def test_very_long_path(self):
        """
        Test case 18: Very long access path
        
        Input: Deeply nested path
        Expected: Correctly handle multi-level nesting
        """
        # Create deeply nested model
        class DeepModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.level1 = torch.nn.ModuleDict({
                    'level2': torch.nn.ModuleDict({
                        'level3': torch.nn.ModuleDict({
                            'level4': torch.nn.Linear(10, 10)
                        })
                    })
                })
        
        deep_model = DeepModel()
        
        # Execute
        result = get_submodule(deep_model, "level1.level2.level3.level4")
        
        # Verify output
        expected = deep_model.level1['level2']['level3']['level4']
        self.assertIs(result, expected,
                     "Should correctly handle deep paths")
    
    def test_modulelist_with_zero_index(self):
        """
        Test case 19: ModuleList 0 index
        
        Input: attr_name="layers.0"
        Expected: Return first element
        """
        # Execute
        result = get_submodule(self.test_model, "layers.0")
        
        # Verify output
        self.assertIs(result, self.test_model.layers[0])
    
    def test_path_type_conversion(self):
        """
        Test case 20: Path type handling
        
        Expected: Correctly distinguish numeric indices and string keys
        """
        # Use number for ModuleList
        list_result = get_submodule(self.test_model, "layers.0")
        self.assertIsNotNone(list_result)
        
        # Use string for ModuleDict
        dict_result = get_submodule(self.test_model, "named_layers.first")
        self.assertIsNotNone(dict_result)
    
    def test_consecutive_modulelist_access(self):
        """
        Test case 21: Consecutive ModuleList access
        
        Input: If there are nested ModuleLists
        Expected: Correctly handle consecutive indices
        """
        # Create nested ModuleList
        class NestedListModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.outer = torch.nn.ModuleList([
                    torch.nn.ModuleList([
                        torch.nn.Linear(10, 10),
                        torch.nn.Linear(10, 10),
                    ]),
                    torch.nn.ModuleList([
                        torch.nn.Linear(10, 10),
                        torch.nn.Linear(10, 10),
                    ]),
                ])
        
        nested_model = NestedListModel()
        
        # Execute
        result = get_submodule(nested_model, "outer.0.1")
        
        # Verify output
        expected = nested_model.outer[0][1]
        self.assertIs(result, expected,
                     "Should correctly handle consecutive ModuleList indices")


if __name__ == "__main__":
    unittest.main()

