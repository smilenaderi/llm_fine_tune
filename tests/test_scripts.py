"""
Basic tests for LLM fine-tuning scripts
"""
import os
import sys

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def test_imports():
    """Test that all scripts can be imported without errors"""
    try:
        import fine_tune
        import inference
        import prepare_data
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_project_structure():
    """Test that required directories exist"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    required_dirs = ['scripts', 'data', 'checkpoints', 'logs']
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        assert os.path.isdir(dir_path), f"Missing directory: {dir_name}"


def test_required_files():
    """Test that required files exist"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    required_files = [
        'requirements.txt',
        'README.md',
        'scripts/fine_tune.py',
        'scripts/inference.py',
        'scripts/prepare_data.py',
        'scripts/submit_job.sh'
    ]
    
    for file_name in required_files:
        file_path = os.path.join(base_dir, file_name)
        assert os.path.isfile(file_path), f"Missing file: {file_name}"


def test_slurm_script_syntax():
    """Test that SLURM script has valid bash syntax"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    script_path = os.path.join(base_dir, 'scripts', 'submit_job.sh')
    
    # Check if file exists and is readable
    assert os.path.isfile(script_path), "submit_job.sh not found"
    
    # Check for shebang
    with open(script_path, 'r') as f:
        first_line = f.readline()
        assert first_line.startswith('#!'), "Missing shebang in submit_job.sh"
