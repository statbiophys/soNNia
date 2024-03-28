from setuptools import setup, find_packages

data_files_to_include = [('', ['README.md', 'LICENSE'])]

setup(name='sonnia',
      version='0.2.0',
      description='Infer and compute selection factors of CDR3 sequences with neural networks.',
      long_description='SoNNia is a python 3.6/2.7 software developed to infer selection pressures on features of amino acid CDR3 sequences. The inference is based on maximizing the likelihood of observing a selected data sample given a representative pre-selected sample. This method was first used in Elhanati et al (2014) to study thymic selection. For this purpose, the pre-selected sample can be generated internally using the OLGA software package, but SoNNia allows it also to be supplied externally, in the same way the data sample is provided. SoNNia takes as input TCR CDR3 amino acid sequences, with or without per sequence lists of possible V and J genes suspected to be used in the recombination process for this sequence. Its output is sequence level selection factors which indicate how more or less represented this sequence would be in the selected pool as compared to the the pre-selected pool. These in turn could be used to calculate the probability to observe any sequence after selection and sample from the selected repertoire. The current package is able to infer non linear selection models via a neural network architecture.',
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
      install_requires=['numpy','tensorflow>=2.1.0','matplotlib','olga>=1.1.3','tqdm'],
      packages = find_packages(),
      package_data = {
            'default_models': [],
            'default_models/human_T_alpha/': ['sonnia/default_models/human_T_alpha/*'],
            'default_models/human_T_beta/': ['sonnia/default_models/human_T_beta/*'],
            'default_models/mouse_T_beta/': ['sonnia/default_models/mouse_T_beta/*'],
            'default_models/human_B_heavy/': ['sonnia/default_models/human_B_heavy/*'],
            'default_models/human_B_kappa/': ['sonnia/default_models/human_B_kappa/*'],
            'default_models/human_B_lambda/': ['sonnia/default_models/human_B_lambda/*'],
            },
      entry_points = {'console_scripts': [
            'sonnia-infer=sonnia.infer:main',
            'sonnia-generate=sonnia.generate:main',
            'sonnia-evaluate=sonnia.evaluate:main'], },
      data_files = data_files_to_include,
      include_package_data=True,
      zip_safe=False)
