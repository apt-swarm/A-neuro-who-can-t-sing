from setuptools import setup

setup(
    name='a-neuro-who-cant-sing',
    version='0.1',
    py_modules=['main'],  # because main.py is a single module, not a package
    install_requires=[
        'pygame',
        'zengl',
        'numpy',
        'pillow',
    ],
    entry_points={
        'console_scripts': [
            'a-neuro-who-cant-sing = entrypoint:main',  # expects a main() function inside main.py
        ]
    },
)