from setuptools import setup, find_packages

setup(
  name = 'alphafold2-pytorch',
  packages = find_packages(),
  version = '0.0.9',
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
    'numpy',
    'torch>=1.6',
    'mdtraj>=1.8',
    'se3-transformer-pytorch'
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
