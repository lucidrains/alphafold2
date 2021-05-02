from setuptools import setup, find_packages

setup(
  name = 'alphafold2-pytorch',
  packages = find_packages(),
  version = '0.0.96',
  license='MIT',
  description = 'AlphaFold2 - Pytorch',
  author = 'Phil Wang, Eric Alcaide',
  author_email = 'lucidrains@gmail.com, ericalcaide1@gmail.com',
  url = 'https://github.com/lucidrains/alphafold2',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'protein folding'
  ],
  install_requires=[
    'einops>=0.3',
    'En-transformer>=0.1.11',
    'mdtraj>=1.8',
    'numpy',
    'performer-pytorch>=1.0.11',
    'proDy',
    'requests',
    'se3-transformer-pytorch>=0.3.9',
    'sidechainnet',
    'torch>=1.6',
    'tqdm',
    'biopython'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
