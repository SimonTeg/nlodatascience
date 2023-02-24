from pathlib import Path
from setuptools import setup  # find_packages,
dependencies = ['numpy', 'pandas', 'tqdm', 'plotly','xgboost',
                'sklearn', 'scipy', 'statsmodels', 'scikit-learn']
# read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name='nlodatascience',
    packages=['nlodatascience'],
    version='0.0.1',
    description='The `nlodatascience` is a package made for NLO to do some data science analyses and visualize them',
    author='Simon Teggelaar',
    author_email='simonteggelaar@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    project_urls={
        "Bug Tracker": "https://github.com/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=dependencies,
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)