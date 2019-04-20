from distutils.core import setup
setup(
  name = 'pfsm',
  packages = ['pfsm'],
  version = '0.1.2',
  license='GNU General Public License v3.0',
  description = 'Python Fast Strings Matching',
  author = 'ngshya',
  author_email = 'ngshya@gmail.com',
  url = 'https://github.com/ngshya/pfsm',
  download_url = 'https://github.com/ngshya/pfsm/archive/v0.1.2.tar.gz',
  keywords = ['strings matching', 'strings distance', 'cosine similarity', 'ngrams', 'tf-idf', 'fast', 'python'],
  install_requires=[
      'unidecode',
      'scikit-learn',
      'numpy', 
      'scipy',
      'sparse_dot_topn',
      'pandas'
      ],
  package_data={
        'notebooks': ['notebooks/*'],
        'data': ['data/*']
  },
  classifiers=[
    'Development Status :: 3 - Alpha', # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" 
    'Intended Audience :: Developers',
    'Topic :: Text Processing',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3.6'
  ],
)

# To upload the package:
# python setup.py sdist
# twine upload dist/*
# rm -R dist
