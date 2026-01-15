from setuptools import setup, find_packages

setup(
    name='exgrpo',
    version='0.0.1',
    description='ExGRPO',
    author='Runzhe Zhan',
    author_email='nlp2ct.runzhe@gmail.com',
    packages=find_packages(include=['deepscaler',]),
    install_requires=[
        'google-cloud-aiplatform',
        'pylatexenc',
        'sentence_transformers',
        'tabulate',
        'math-verify[antlr4_9_3]==0.6.0',
        'flash_attn==2.7.4.post1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
