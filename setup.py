from setuptools import setup, find_packages

data_files_to_include = [('', ['README.md', 'LICENSE'])]

setup(name='sonnia',
      version='0.0.4',
      description='Infer and compute selection factors of CDR3 sequences',
      long_description='Some extensions to sonia',
      url='https://github.com/statbiophys/soNNia',
      author='Giulio Isacchini',
      author_email='giulioisac@gmail.com',
      license='GPLv3',
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.6',
            ],
      packages=find_packages(),
      install_requires=['numpy','tensorflow>=2.1.0','matplotlib','olga>=1.1.3','tqdm','sonia'],
      entry_points = {'console_scripts': [
            'sonnia-infer=sonnia.infer:main',
            'sonnia-generate=sonnia.generate:main',
            'sonnia-evaluate=sonnia.evaluate:main'], },
      data_files = data_files_to_include,
      include_package_data=True,
      zip_safe=False)
