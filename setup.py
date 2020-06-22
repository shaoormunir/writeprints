from distutils.core import setup
setup(
  name = 'writeprints',
  packages = ['writeprints'],
  version = '0.1',
  license='MIT',
  description = 'This package extracts writeprints features from a text document',
  author = 'Shaoor Munir',
  author_email = 'shaoormunir@outlook.com',
  url = 'https://github.com/shaoormunir/writeprints',
  download_url = 'https://github.com/shaoormunir/writeprints/archive/v0.1.tar.gz',
  keywords = ['NLP', 'Machine Leanrning', 'Natural Language Processing', 'Text Features'],
  install_requires=[
          'nltk',
          'spacy',
          'numpy',
          'sortedcontainers',
          'keras',
          'pandas',
          'tensorflow',
          'tqdm',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)