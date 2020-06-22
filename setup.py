from distutils.core import setup
setup(
    name='writeprints',
    packages=['writeprints'],
    version='0.1.1',
    license='MIT',
    include_package_data=True,
    description='This package extracts writeprints features from a text document',
    author='Shaoor Munir',
    author_email='shaoormunir@outlook.com',
    url='https://github.com/shaoormunir/writeprints',
    download_url='https://github.com/shaoormunir/writeprints/archive/v0.1.1.tar.gz',
    keywords=['NLP', 'Machine Leanrning',
              'Natural Language Processing', 'Text Features'],
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
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt"],
    },
)